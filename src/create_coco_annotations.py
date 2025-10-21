# -*- coding:utf-8 -*-
# Create coco-like annotations for the given images and captions


import os
import sys
import argparse
import json
from tqdm import tqdm

flickr8k_cap_fn = "/root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt"
flickr8k_val_dir = "/root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_val"

annotations = []

img_files = os.listdir(flickr8k_val_dir)

image_idx_dict = {}
image_idx = 0
image_cap_list = {}

with open(flickr8k_cap_fn, "r") as fin:
    lines = fin.readlines()

lines = lines[1:]
for i, line in tqdm(enumerate(lines), total = len(lines)):
    image_name, caption = line.strip().split(",", 1)
    image_idx_dict[image_name] = image_idx
    if image_idx not in image_cap_list:
        image_cap_list[image_idx] = []
    image_cap_list[image_idx].append(i+1)
    annotation = {}
    annotation["image_id"] = image_idx
    annotation["id"] = i + 1
    annotation["caption"] = caption
    annotations.append(annotation)
    image_idx += 1

info_dict = {    
    "description": "Flickr 8K Dataset",
    "url": "None",
    "version": "1.0",
    "year": 2025,
    "contributor": "Flickr",
    "date_created": "2025/01/16",
}

caption_dict = {
    "info": info_dict,
    "images": [
        {
            "license": 3,
            "file_name": img_name,
            "id": image_idx_dict[img_name],
        }
        for img_name in image_idx_dict.keys()
    ],
    "annotations": annotations,
}

with open("/root/autodl-tmp/datasets/image_caption_generation/flickr8k/annotations/captions_val.json", "w") as fout:
    json.dump(caption_dict, fout)
