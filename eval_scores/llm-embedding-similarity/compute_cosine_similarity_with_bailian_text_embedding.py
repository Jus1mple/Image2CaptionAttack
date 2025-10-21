# -*- coding:utf-8 -*-
# Use the pre-trained LLM model to map the generated caption and the ground truth captions into an embedding space, and calculate the cosine similarities between the generated caption and each ground truth caption.

import os
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
import argparse
import json
from tqdm import tqdm


api_key = "" # SETUP YOUR OWN API KEY
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


parser = argparse.ArgumentParser()

parser.add_argument("--result_root_dir", type = str, default = "/root/Image2CaptionAttack/results")
parser.add_argument("--dataset-name", type = str, default = "COCO2017")
parser.add_argument("--blip-model", type = str, default = "blip2-opt-2.7b")
parser.add_argument("--clip-model", type = str, default = "ViT-16B")
parser.add_argument("--leaked-feature-layer", type = str, default = "vit-base")
parser.add_argument("--out_result_dir", type = str, default = "/root/Image2CaptionAttack/eval_scores/llm-embedding-similarity/scores")

args = parser.parse_args()

result_root_dir = args.result_root_dir
dst_name = args.dataset_name
blip_model = args.blip_model
clip_model = args.clip_model
leaked_feature_layer = args.leaked_feature_layer
out_result_dir = args.out_result_dir

result_fn = os.path.join(
    result_root_dir,
    f"{dst_name}_{blip_model}_{clip_model}_{leaked_feature_layer}",
    "eval_results.jsonl",
)

data = []
with open(result_fn, 'r') as fin:
    for line in fin:
        line = line.strip('\n').strip('\r')
        line = json.loads(line)
        data.append(line)

client = OpenAI(
    api_key = api_key, 
    base_url = base_url,
)

def get_embedding_from_aliyun(text):
    response = client.embeddings.create(
        model="text-embedding-v1",
        input=text,
        dimensions=1024,
        encoding_format="float",
    )
    return response.data[0].embedding


def clean_generated_caption(caption):
    caption = caption.replace("\ufffd", "")
    caption = caption.replace("\u200b", "")
    caption = caption.replace("\u200d", "")
    caption = caption.replace("\u200c", "")
    caption = caption.replace("\u200e", "")
    caption = caption.strip(" ")
    caption = caption.replace("dfx", "")
    caption = caption.replace("  ", " ")
    return caption


out_result_fn = os.path.join(out_result_dir, f"eval_cosine_similarity_{dst_name}_{blip_model}_{clip_model}_{leaked_feature_layer}.jsonl")
fout = open(out_result_fn, 'w')

for i, item in tqdm(enumerate(data), total = len(data), desc = "Computing cosine similarity"):
    generated_caption = item["generated"]
    # clean generated caption
    generated_caption = clean_generated_caption(generated_caption)
    ground_truth_captions = item["ground_truth"]
    generated_cap_embedding = get_embedding_from_aliyun(generated_caption)
    ground_truth_cap_embeddings = [get_embedding_from_aliyun(caption) for caption in ground_truth_captions]
    cosine_sims = [
        F.cosine_similarity(
            torch.tensor(generated_cap_embedding, device="cuda:0"),
            torch.tensor(ground_truth_cap_embedding, device="cuda:0"),
            dim=0,
        ).cpu().item()
        for ground_truth_cap_embedding in ground_truth_cap_embeddings
    ]
    item["cosine_sims"] = cosine_sims
    item["mean_cosine_sim"] = np.mean(cosine_sims)
    # item["median_cosine_sim"] = np.median(cosine_sims)
    item["max_cosine_sim"] = np.max(cosine_sims)
    item["min_cosine_sim"] = np.min(cosine_sims)

    fout.write(json.dumps(item) + '\n')

fout.close()
