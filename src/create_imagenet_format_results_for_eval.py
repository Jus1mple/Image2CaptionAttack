# -*- coding: utf-8 -*-


import os
import sys
import json
import argparse
import json

parser = argparse.ArgumentParser(description="Create format results for evaluation")

parser.add_argument(
    "--cap-file",
    type=str,
    default="/root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test/caption_test.json",
    # default=r"H:\Code\AutoDL_backup\datasets\image_caption_generation\flickr8k\annotations\captions_val.json",
)
parser.add_argument(
    "--result_root_dir", type=str, default="/root/Image2CaptionAttack/results"
)
parser.add_argument("--dataset-name", type=str, default="flickr8k")
parser.add_argument("--blip2-model", type=str, default="blip2-opt-2.7b")
parser.add_argument("--victim-model", type=str, default="ViT-32B")
parser.add_argument("--leaked-feature-layer", type=str, default="vit-base")
parser.add_argument("--add-noise", type = str, default = "False")
parser.add_argument(
    "--out_result_dir", type=str, default="/root/Image2CaptionAttack/processed_results"
)
args = parser.parse_args()

cap_file = args.cap_file
result_root_dir = args.result_root_dir
dataset_name = args.dataset_name
blip2_model = args.blip2_model
victim_model = args.victim_model
leaked_feature_layer = args.leaked_feature_layer
add_noise = True if args.add_noise == "True" else False
out_result_dir = args.out_result_dir

result_dir = os.path.join(
    result_root_dir,
    "{}_{}_{}_{}".format(dataset_name, blip2_model, victim_model, leaked_feature_layer),
)
# out_result_dir = os.path.join(out_result_dir, "{}_{}_{}".format(blip2_model, clip_model, leaked_feature_layer))

with open(cap_file, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
    annotations = json.load(fin)

images_list = annotations["images"]
image_idx_dict = {
    image["file_name"] : image["id"] for image in images_list
}

eval_results_file = os.path.join(result_dir, "eval_results.jsonl" if not add_noise else "eval_results_noise.jsonl")
out_fn = os.path.join(
    out_result_dir,
    "results_{}_{}_{}_{}.json".format(
        dataset_name, blip2_model, victim_model, leaked_feature_layer
    ) if not add_noise else "results_{}_{}_{}_{}_noise.json".format(
        dataset_name, blip2_model, victim_model, leaked_feature_layer
    ),
)

results = []
with open(eval_results_file, "r") as fin:
    for line in fin:
        line = line.strip("\n").strip("\r")
        line = json.loads(line)

        results.append(
            {
                "image_id": int(image_idx_dict[line["image_id"]]),
                "caption": line["generated"],
                "ground_truth_captions": line["ground_truth"],
            }
        )

with open(out_fn, "w") as fout:
    json.dump(results, fout, indent=4)
