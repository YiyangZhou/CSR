import os
import re
import json
import argparse
from utils import load_pickles, Rank_Node


def dfs(node, path=[], cumulative_score=0, clip_alpha=0.8):
    if node.rank is not None and node.clip_rank is not None:
        cumulative_score += (1-clip_alpha)*node.rank + clip_alpha * node.clip_rank
    current_path = path + [(node.text, cumulative_score)]
    if node.is_final:
        return [(current_path, cumulative_score)]
    paths_scores = []

    for child in node.children:
        paths_scores.extend(dfs(child, current_path, cumulative_score, clip_alpha))
    return paths_scores


def process_data(args):
    folder_path = args.folder_path
    image_dir = args.image_dir
    clip_alpha = args.clip_alpha
    output_file = args.output_file
    tree_list = load_pickles(folder_path)
    data_list = []
    data_list_with_score = []

    for tree_dict in tree_list:
        this_id_dict = {}
        qid = tree_dict['qid']
        tree = tree_dict['tree']

        tree.calculate_ranks()

        img_path = str(qid)+'.jpg'
        results = dfs(tree, clip_alpha=clip_alpha)
        sorted_results = sorted(results, key=lambda x: x[1] / len(x[0]))
        chosen_process = sorted_results[0][0]
        rejected_process = sorted_results[-1][0]

        the_input = chosen_process[0][0].strip()
        pattern = r"USER:\s*<image>\s*"
        replacement = "USER: <image>"

        chosen = re.sub(pattern, replacement, chosen_process[-1][0])
        rejected = re.sub(pattern, replacement, rejected_process[-1][0])
        chosen = chosen[len(the_input):].strip()
        rejected = rejected[len(the_input):].strip()

        chosen_conv = [{'from': 'human', 'value': the_input}, {'from': 'gpt', 'value': chosen}]
        rejected_conv = [{'from': 'human', 'value': the_input}, {'from': 'gpt', 'value': rejected}]

        this_id_dict['id'] = qid
        this_id_dict['image'] = os.path.join(image_dir, 'COCO_train2014_' + img_path)
        this_id_dict['conversations'] = chosen_conv
        this_id_dict['rejected_conversations'] = rejected_conv

        data_list.append(this_id_dict)
        data_list_with_score.append((this_id_dict, chosen_process[-1][1] - rejected_process[-1][1]))

    with open(output_file, mode='w') as json_file:
        json.dump(data_list, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Directory to save step2's .pkl results")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--clip_alpha", type=float, default=0.9, help="Alpha value for CLIP")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output CSR JSON dataset")
    args = parser.parse_args()

    process_data(args)
