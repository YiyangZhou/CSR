from utils import *
from accelerate.utils import gather_object
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoTokenizer
import json
from accelerate import Accelerator
from PIL import Image
import os
import argparse


def load_hf_llava_model(model_path):
    # Weights are loaded directly with hf-llava version
    model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map='cpu', torch_dtype=torch.float16)
    model_tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side='left') 
    return model, model_tokenizer, base_tokenizer


def load_llava_model(model_path, base_hf_model_path, mapping_path):
    # Weights should be specially loaded with other llava versions
    model = LlavaForConditionalGeneration.from_pretrained(base_hf_model_path, device_map='cpu', torch_dtype=torch.float16)
    model_tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_tokenizer = AutoTokenizer.from_pretrained(base_hf_model_path, use_fast=False, padding_side='left')
    state_dicts = load_and_merge_models(model_path)
    with open(mapping_path, 'r', encoding='utf-8') as f1:
        mapping_keys = json.load(f1)

    modified_weights = {}
    for old_key, value in state_dicts.items():
        new_key = mapping_keys.get(old_key, old_key)
        modified_weights[new_key] = value
    modified_weights['language_model.model.embed_tokens.weight'] = model.state_dict()['language_model.model.embed_tokens.weight']
    modified_weights['language_model.lm_head.weight'] = model.state_dict()['language_model.lm_head.weight']
    model.load_state_dict(modified_weights, strict=True)
    return model, model_tokenizer, base_tokenizer


def sentence_level_beam_search_tree(qid, model, accelerator, processor, tokenizer, after_tokenizer, initial_text, images, sentence_end_id, max_length, max_new_tokens, num_beams, num_beam_group, token_level_beams, diversity_penalty):
    root = Node(initial_text, 0, 0)
    active_nodes = [root]

    with torch.no_grad():
        while active_nodes:
            new_nodes = []

            for node in active_nodes:
                inputs = processor(text=node.text, images=images, return_tensors="pt").to(model.device)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        num_beams=token_level_beams,
                        eos_token_id=sentence_end_id,
                        num_beam_groups=num_beam_group,
                        diversity_penalty=diversity_penalty,
                        pad_token_id=tokenizer.pad_token_id,
                        num_return_sequences=token_level_beams,
                        max_new_tokens=max_new_tokens,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                gen_sequences = outputs.sequences[:, inputs.input_ids.shape[-1]:]
                gen_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                for j, (text, score) in enumerate(zip(gen_texts, outputs.sequences_scores)):
                    new_score = node.score + score.item()
                    is_final = (tokenizer.eos_token_id in gen_sequences[j].tolist()) or (after_tokenizer.eos_token_id in gen_sequences[j].tolist() or len(tokenizer.decode(outputs.sequences[j])) >= max_length)
                    new_node = Node(text, new_score, node.depth + 1, node, is_final)
                    node.add_child(new_node)

                    if not is_final:
                        new_nodes.append(new_node)

            new_nodes.sort(key=lambda x: x.score, reverse=True)
            active_nodes = new_nodes[:int(num_beams/2)-1] + new_nodes[-int(num_beams/2):] if len(new_nodes) >= num_beams else new_nodes

            if not active_nodes:
                break

    return [{'id': qid, 'tree': root}]

def eval_model(args):
    accelerator = Accelerator()
    model_path = args.model_path
    base_hf_model_path = args.base_hf_model_path
    mapping_path = args.weight_mapping_path
    output_dir = args.output_dir

    # Load Model
    processor = AutoProcessor.from_pretrained(base_hf_model_path)
    if args.is_hf:
        model, model_tokenizer, base_tokenizer = load_hf_llava_model(model_path)
    else:
        model, model_tokenizer, base_tokenizer = load_llava_model(model_path, base_hf_model_path, mapping_path)
    model.to(accelerator.device)

    # Load Dataset
    with open(args.dataset_path, 'r', encoding='utf8') as fp:
        my_dataset = json.load(fp)
    llava_loader = get_llava_dataloader(my_dataset, 1)
    llava_loader, processor = accelerator.prepare(llava_loader, processor)

    with torch.no_grad():
        for data in llava_loader:
            input_questions = data['input']
            input_questions = [q.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "") for q in input_questions]
            image_paths = data['image']
            qid = data['question_ids']
            images = []

            for image_path in image_paths:
                images.append(Image.open(os.path.join(args.images_dir, 'COCO_train2014_' + image_path)))

            prompts = get_prompts(input_questions)
            sentence_end_id = int(args.period_id)
            max_length = int(args.max_length)
            token_level_beams = int(args.num_token_beams)
            max_new_tokens = int(args.max_new_tokens)
            diversity_penalty = float(args.diversity_penalty)
            num_beams = int(args.num_beams)
            num_beam_group = int(args.num_beam_group)

            # Batched inference is not supported yet
            result = gather_object(sentence_level_beam_search_tree(
                qid[0],
                model,
                accelerator,
                processor,
                base_tokenizer,
                model_tokenizer,
                prompts[0],
                images[0],
                sentence_end_id,
                max_length,
                max_new_tokens,
                num_beams,
                num_beam_group,
                token_level_beams,
                diversity_penalty
            ))

            if accelerator.is_main_process:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                for obj in result:
                    save_path = os.path.join(output_dir, str(obj['id']) + '.pkl')
                    save_object(obj, save_path)

            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='llava-hf/llava-1.5-7b-hf', help="Path to your model")
    parser.add_argument("--base_hf_model_path", type=str, default='llava-hf/llava-1.5-7b-hf', help="Path to huggingface base model")
    parser.add_argument("--is_hf", type=int, default=1, help="If it's a hf model")
    parser.add_argument("--dataset_path", type=str, default='./data/CSR-Prompt-Dataset-12k.json', help="Path to the prompt dataset")
    parser.add_argument("--images_dir", type=str, default="./data/images/train2014", help="Directory to images")
    parser.add_argument("--output_dir", type=str, default="./outputs/sample", help="Path to step1's result")
    parser.add_argument("--diversity_penalty", type=float, default=3.0)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_beam_group", type=int, default=5)
    parser.add_argument("--num_token_beams", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=70)
    parser.add_argument("--period_id", type=int, default=29889)
    parser.add_argument("--weight_mapping_path", type=str, default='./model_mapping/key_mapping_hf_7b.json', help="To load non-hf model specially")
    args = parser.parse_args()

    eval_model(args)
