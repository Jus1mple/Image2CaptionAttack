# -*- coding:utf-8 -*-
# ImageNet-1k Dataset
# I utilize Qwen to generate captions for images in ImageNet-1k dataset.
# I store the generated caption in *.jsonl file format an the original images are stored in *.parquet file format.
# I randomly select 300 images from each parquet file and generate captions for them.
#


import os
import sys
import json
import torch
import pyarrow.parquet as pq
import random
import cv2
import numpy as np
torch.backends.cudnn.enabled = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.multiprocessing.set_start_method('spawn')

sys.path.append(os.path.join(os.getcwd(), ".."))

from utils import load_victim_model
from NoiseResNet import NoiseResNetCLIP


class ImageNetCapProcessor:

    def __init__(
        self,
        data_dir,  # only specify the root directory,
        annotation_file=None,  # useless for this dataset
        caption_file=None,  # useless for this dataset
        target_size=224,
        normalize=True,
        clamp_output=False,
        victim_model="ViT-B/32",
        victim_device="cuda:0",
        extract_feature_func=None,
        add_noise=False,
    ):
        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.caption_file = caption_file
        self.target_size = target_size
        self.normalize = normalize
        self.clamp_output = clamp_output
        self.add_noise = add_noise
        # self.clip_model = clip_model
        self.victim_device = victim_device
        self.victim_model = load_victim_model(victim_model, victim_device)
        self.victim_model = self.victim_model.to(torch.float)
        if self.add_noise:
            self.victim_model = NoiseResNetCLIP(self.victim_model)
        self.extract_feature_func = extract_feature_func
        self.add_noise = add_noise

        files = os.listdir(data_dir)
        self.parquet_files = [f for f in files if f.endswith(".parquet")]
        self.caption_files = [f for f in files if f.endswith(".jsonl")]
        self.data = self.process_all_data()
        self.image_names = list(self.data.keys())
        self.image_ids = list(range(len(self.image_names)))

    def process_all_data(self):
        caption_filename_list = [cap_fn.split(".")[0] for cap_fn in self.caption_files]
        data_dict = {}
        for cap_fn in caption_filename_list:
            # print(cap_fn)
            pq_fn = cap_fn.split("_annotations")[0]
            pq_file = os.path.join(self.data_dir, f"{pq_fn}.parquet")
            cap_file = os.path.join(self.data_dir, f"{cap_fn}.jsonl")
            # cap_dict = {}
            with open(cap_file, "r") as fin:
                for line in fin:
                    line = line.strip('\n').strip('\r')
                    line = json.loads(line)
                    data_dict[line["image_path"]] = {
                        "image_label": line["image_label"],
                        "caption": [line["image_caption"]],
                    }
            table = pq.read_table(pq_file)
            pq_dict = table.to_pydict()
            for image in pq_dict["image"]:
                if image["path"] not in data_dict:
                    continue
                img_bytes = image["bytes"]
                data_dict[image["path"]]["bytes"] = img_bytes
        return data_dict

    def get_image_name(self, image_id):
        return self.image_names[image_id]

    def preprocess_image_with_idx(self, image_idx):
        image_name = self.image_names[image_idx]
        image_bytes = self.data[image_name]["bytes"]
        image_label = self.data[image_name]["image_label"]
        image_caption = self.data[image_name]["caption"]
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_size, self.target_size))
        if self.normalize:
            image = cv2.normalize(
                image,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )

        image_tensor = torch.tensor(image, dtype = torch.float).permute(2, 0, 1)
        if self.clamp_output:
            image_tensor = torch.clamp(image_tensor, 0., 1.)
        return image_tensor, image_caption

    def preprocess_image_with_name(self, image_name):
        image_bytes = self.data[image_name]["bytes"]
        image_label = self.data[image_name]["image_label"]
        image_caption = self.data[image_name]["caption"]
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_size, self.target_size))
        if self.normalize:
            image = cv2.normalize(
                image,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )

        image_tensor = torch.tensor(image, dtype = torch.float).permute(2, 0, 1)
        if self.clamp_output:
            image_tensor = torch.clamp(image_tensor, 0., 1.)
        return image_tensor, image_caption

    def extract_clip_features(self, image_tensor):
        if image_tensor.dim() == 3:  # Add batch dimension if missing
            image_tensor = image_tensor.unsqueeze(0)
        if not self.extract_feature_func:
            clip_img_features = self.victim_model.encode_image(
                image_tensor.to(self.victim_device)
            )
        else:
            clip_img_features = self.extract_feature_func(
                image_tensor.to(self.victim_device), self.victim_model
            )
        return clip_img_features.to(torch.float).squeeze(0)

    def process_image_and_caption_with_idx(self, image_idx):
        image_name = self.image_names[image_idx]
        image_tensor, caption = self.preprocess_image_with_idx(image_idx)
        image_features = self.extract_clip_features(image_tensor)

        return image_name, image_tensor, image_features, caption

    def get_data_loader(
        self, 
        batch_size = 32, 
        max_samples = None, 
        shuffle = True, 
        collate_fn = None
    ):
        from torch.utils.data import Dataset, DataLoader

        class ImageNetCapDataLoader(Dataset):
            def __init__(self, processor, shuffle=False, max_samples=None):
                self.processor = processor
                self.image_names = processor.image_names
                self.image_ids = processor.image_ids
                if shuffle:
                    random.shuffle(self.image_ids)
                if max_samples and max_samples > 0 and max_samples < len(self.image_ids):
                    self.image_ids = self.image_ids[:max_samples]
                self.image_ids = self.image_ids
                self.image_names = [self.image_names[i] for i in self.image_ids]

            def __len__(self):
                return len(self.image_ids)

            def __getitem__(self, idx):
                image_idx = self.image_ids[idx]
                image_name, image_tensor, image_features, caption = (
                    self.processor.process_image_and_caption_with_idx(image_idx)
                )
                return image_name, image_tensor, image_features, caption

        imagenet_dataset = ImageNetCapDataLoader(self, shuffle = shuffle, max_samples = max_samples)
        return DataLoader(
            imagenet_dataset, 
            batch_size= batch_size, 
            collate_fn=collate_fn
        )

    def __len__(self):
        return len(self.image_names)


if __name__ == "__main__":
    data_dir = "/root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train"
    processor = ImageNetCapProcessor(data_dir)
    dataloader = processor.get_data_loader(batch_size = 32, max_samples = 900, shuffle = True)
    print(len(processor))
    for batch in dataloader:
        print(batch)
        break
