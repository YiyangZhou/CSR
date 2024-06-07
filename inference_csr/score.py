import os
import json
from transformers import CLIPModel, AutoProcessor
import torch
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import Node, Rank_Node, extract_new_text, load_and_store_pkl_files, clean_tree, save_pickle


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    return batch


def get_clip_score(new_text, image, model, processor):
    if not new_text:
        return None
    inputs = processor(text=[new_text], images=image, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    clip_score = logits_per_image.cpu().detach().numpy()[0][0]
    return clip_score


def dfs_score(node, model, processor, parent=None, image=None):
    if image is None:
        raise ValueError("Image must be provided")

    new_text = extract_new_text(node.text, parent.text if parent else None)
    clip_score = get_clip_score(new_text, image, model, processor) if parent else None

    rank_node = Rank_Node(
        text=node.text,
        score=node.score,
        depth=node.depth,
        parent=parent,
        is_final=node.is_final,
        clip_score=clip_score
    )

    if parent:
        parent.add_child(rank_node)

    for child in node.children:
        child_len = len(extract_new_text(child.text, node.text))
        if child_len >= 4:
            dfs_score(child, model, processor, rank_node, image)

    return rank_node


def get_result(qid, tree, clip_model, clip_processor, image):
    new_tree = dfs_score(tree, clip_model, clip_processor, None, image=image)
    new_tree.calculate_ranks()
    return [{'qid': qid, 'tree': new_tree}]


def eval_model(args):
    folder_path = args.folder_path
    pkl_data_list = load_and_store_pkl_files(folder_path)
    output_dir = args.output_dir

    with open(args.data_json, 'r') as file:
        data = json.load(file)

    id_image_map = {item['id']: item['image'] for item in data}

    list_dataset = ListDataset(pkl_data_list)
    dataloader = DataLoader(list_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    image_dir = args.image_dir
    clip_model = CLIPModel.from_pretrained(args.clip_model_path)
    clip_processor = AutoProcessor.from_pretrained(args.clip_model_path)
    accelerator = Accelerator()
    clip_model, clip_processor, dataloader = accelerator.prepare(clip_model, clip_processor, dataloader)

    for tree_dict in dataloader:
        tree_dict = tree_dict[0]
        qid = tree_dict['id']
        tree = clean_tree(tree_dict['tree'])
        img_path = id_image_map[qid]
        image = Image.open(os.path.join(image_dir, 'COCO_train2014_' + img_path))
        with torch.no_grad():
            result = gather_object(get_result(qid, tree, clip_model, clip_processor, image))

        if accelerator.is_main_process:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for obj in result:
                save_path = os.path.join(output_dir, str(obj['qid']) + '.pkl')
                save_pickle(obj, save_path)

        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Directory to the step1's .pkl results")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save step2's .pkl results")
    parser.add_argument("--data_json", type=str, required=True, help="Path to the JSON data file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--clip_model_path", type=str, default='openai/clip-vit-large-patch14-336', help="Path to the CLIP model")
    args = parser.parse_args()

    eval_model(args)
