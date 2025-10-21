# -*- coding:utf-8 -*-
# Create coco-like annotations for the given images and captions


import os
import sys
import argparse
import json
from tqdm import tqdm

# flickr8k_cap_fn = (
#     "/root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt"
# )

imagenet_test_annotation_files = [
    "/root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test/test-00000-of-00003_annotations.jsonl",
    "/root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test/test-00001-of-00003_annotations.jsonl",
    "/root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test/test-00002-of-00003_annotations.jsonl",
]

annotations = []

image_idx_dict = {}
image_idx = 0
image_cap_list = {}

for imagenet_test_annotation_file in imagenet_test_annotation_files:
    with open(imagenet_test_annotation_file, 'r') as fin:
        lines = fin.readlines()
    
    for i, line in tqdm(enumerate(lines), total=len(lines)):
        line = line.strip('\n').strip('\r')
        line = json.loads(line)
        
        image_name = line["image_path"]
        caption = line["image_caption"]
        image_idx_dict[image_name] = image_idx
        if image_idx not in image_cap_list:
            image_cap_list[image_idx] = []
        image_cap_list[image_idx].append(i + 1)
        annotation = {}
        annotation["image_id"] = image_idx
        annotation["id"] = i + 1
        annotation["caption"] = caption
        annotations.append(annotation)
        image_idx += 1


info_dict = {
    "description": "ImageNet-1K Dataset",
    "url": "None",
    "version": "1.0",
    "year": 2025,
    "contributor": "ImageNet",
    "date_created": "2025/02/05",
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

with open(
    "/root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test/caption_test.json",
    "w",
) as fout:
    json.dump(caption_dict, fout)
