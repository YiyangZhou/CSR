from transformers import StoppingCriteria
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader
import dataclasses
from enum import auto, Enum
from typing import List
import base64
from io import BytesIO
import pickle
import glob
import re

DEFAULT_IMAGE_TOKEN = "<image>"

def get_safe_tensor(sf_path):
    from safetensors import safe_open

    tensors = {}
    with safe_open(sf_path, framework="pt", device='cpu') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

def load_and_merge_models(model_folder_path):
    merged_model_state_dict = {}
    for model_file in os.listdir(model_folder_path):
        if model_file.endswith('.bin'):
            file_path = os.path.join(model_folder_path, model_file)
            model_state_dict = torch.load(file_path, map_location='cpu')
            for key, value in model_state_dict.items():
                if key not in merged_model_state_dict:
                    merged_model_state_dict[key] = value
        elif model_file.endswith('.safetensors'):
            file_path = os.path.join(model_folder_path, model_file)
            model_state_dict = get_safe_tensor(file_path)
            for key, value in model_state_dict.items():
                if key not in merged_model_state_dict:
                    merged_model_state_dict[key] = value
    return merged_model_state_dict


def get_done_ids(file_path):
    id_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if 'id' in data:
                id_list.append(data['id'])
    return id_list

def save_object(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def get_prompts(inputs):
    input_questions = [DEFAULT_IMAGE_TOKEN + '\n' + input_question for input_question in inputs]
    prompts = []
    for input_q in input_questions:
        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], input_q)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())
    return prompts


def get_file_names(directory):
    file_names = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path):
            file_names.append(item)
    return file_names


def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data successfully saved to {file_path}")


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def score_to_ranking_score(input_list):
    sorted_list = sorted((e, i) for i, e in enumerate(input_list))
    scores = [0] * len(input_list)
    for rank, (value, original_index) in enumerate(sorted_list, 1):
        scores[original_index] = rank
    return scores


def load_pickles(folder_path):
    pickle_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                pickle_list.append(data)
    return pickle_list


def score_to_ranking_score(input_list):
    sorted_list = sorted((e, i) for i, e in enumerate(input_list))
    scores = [0] * len(input_list)
    for rank, (value, original_index) in enumerate(sorted_list, 1):
        scores[original_index] = rank
    return scores


def load_and_store_pkl_files(folder_path):
    pkl_data_list = []
    for file_path in glob.glob(os.path.join(folder_path, '*.pkl')):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            pkl_data_list.append(data)
    return pkl_data_list


def clean_text(input_text):
    pattern = r'[^\w\s.,?!;:\'\"-()/\[\]{}+$€£*=/><==!%°^™©®♫♪π√]+'
    cleaned_text = re.sub(pattern, '', input_text)
    return cleaned_text


def extract_new_text(current_text, parent_text):
    if parent_text:
        processed_text = current_text[len(parent_text)+3:].strip()
        return clean_text(processed_text)
    return clean_text(current_text)


def clean_tree(node):
    korean_regex = re.compile(r'[\u3130-\u318F\uAC00-\uD7AF]')
    emoji_regex = re.compile(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]'
    )
    cyrillic_regex = re.compile(r'[\u0400-\u04FF]')  # Russian and other Cyrillic characters range
    japanese_regex = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')  # Japanese Hiragana and Katakana range
    nbsp_regex = re.compile(r'(&nbsp;){2,}')  # Matches two or more consecutive &nbsp;

    def contains_special_chars(text):
        """Check if the text contains Korean, emoji, Cyrillic, Japanese characters, or consecutive &nbsp;."""
        return (korean_regex.search(text) is not None or
                emoji_regex.search(text) is not None or
                cyrillic_regex.search(text) is not None or
                japanese_regex.search(text) is not None or
                nbsp_regex.search(text) is not None)

    def contains_repeated_phrases(text, min_repeats=6):
        """Check if the text contains fully consecutive and adjacent repeated phrases."""
        pattern = re.compile(r"(\b\w+\b)(?:\s+\1){" + str(min_repeats - 1) + ",}")
        pattern2 = 'ttttttttttt'
        pattern3 = 'sssssssssss'
        pattern4 = ', , , , , , , , , , , , ,'
        pattern5 = '\'\'\'\'\'\'\'\'\'\''
        pattern6 = re.compile(r"(.)\s*\1\s*\1\s*\1\s*\1\s*\1\s*\1\s*\1\s*\1\s*\1\s*\1")
        return (pattern.search(text) is not None) or (pattern6.search(text) is not None) or (pattern2 in text)  or (pattern3 in text) or (pattern4 in text) or (pattern5 in text)

    def exceeds_newline_limit(text, limit=5):
        return text.count('\n') > limit

    new_children = []
    for child in node.children:
        modified_text = extract_new_text(child.text, node.text)
        contains_special = contains_special_chars(modified_text)
        contains_repetition = contains_repeated_phrases(modified_text)
        exceeds_lines = exceeds_newline_limit(modified_text)

        if not contains_special and not contains_repetition and not exceeds_lines:
            clean_tree(child)
            new_children.append(child)
    node.children = new_children
    return node


class Node:
    def __init__(self, text, score, depth, parent=None, is_final=False):
        self.text = text
        self.score = score
        self.depth = depth
        self.parent = parent
        self.children = []
        self.is_final = is_final

    def add_child(self, child):
        self.children.append(child)


class Rank_Node:
    def __init__(self, text, score, depth, parent=None, is_final=False, clip_score=None):
        self.text = text
        self.score = score
        self.depth = depth
        self.parent = parent
        self.children = []
        self.is_final = is_final
        self.rank = None
        self.clip_score = clip_score
        self.clip_rank = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def calculate_ranks(self):
        if self.children:
            sorted_children_by_score = sorted(self.children, key=lambda x: x.score, reverse=True)
            total_children = len(sorted_children_by_score)
            for index, child in enumerate(sorted_children_by_score):
                child.rank = (index + 1) / total_children

            sorted_children_by_clip_score = sorted(self.children, key=lambda x: x.clip_score, reverse=True)
            for index, child in enumerate(sorted_children_by_clip_score):
                child.clip_rank = (index + 1) / total_children
        for child in self.children:
            child.calculate_ranks()


class LLaVaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # this_image=Image.open(os.path.join('/home/wf/Projects/chengdongjie/LlaVa-Instruct-150K/data/train2014','COCO_train2014_'+self.data[idx]['image']))
        return {'input':self.data[idx]['input'], 'image':self.data[idx]['image'], 'id':self.data[idx]['id']}
    
def collate_fn(batch):
    return{
        "input": [item["input"] for item in batch],
        "image": [item["image"] for item in batch],
        "question_ids": [item["id"] for item in batch]
    }

def get_llava_dataloader(data, batch_size):
    dataset = LLaVaDataset(data)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,

    "mpt": conv_mpt,
}