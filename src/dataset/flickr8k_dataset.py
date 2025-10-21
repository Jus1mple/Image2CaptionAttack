# -*- coding:utf-8 -*-
# Build Flikr8k Dataset Wrapper


import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import clip
import random

from utils import load_victim_model
from NoiseResNet import NoiseResNetCLIP


class Flickr8kProcessor:

    def __init__(
        self,
        data_dir,
        caption_file=None,
        target_size=224,
        normalize=True,
        clamp_output=False,
        victim_model="ViT-B/32",
        victim_device="cuda:0",
        extract_feature_func=None,
        annotation_file=None,  # useless for this dataset
        add_noise=False,
    ):
        self.data_dir = data_dir
        self.target_size = target_size
        self.normalize = normalize
        self.clamp_output = clamp_output
        self.add_noise = add_noise

        # self.victim_model, self.clip_processor = clip.load(victim_model, device = clip_device)
        self.victim_device = victim_device
        self.victim_model = load_victim_model(victim_model, victim_device)
        self.victim_model = self.victim_model.to(torch.float)
        if self.add_noise:
            self.victim_model = NoiseResNetCLIP(self.victim_model)
        self.extract_feature_func = extract_feature_func

        self.image_names = os.listdir(data_dir)

        self.captions = self.load_caps(caption_file)

    def process_one_image(self, image_path):
        image = cv2.imread(image_path)
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

        image_tensor = torch.tensor(image, dtype = torch.float).permute(2, 0, 1) # C, H, W

        if self.clamp_output:
            image_tensor = torch.clamp(image_tensor, 0., 1.)

        return image_tensor

    def load_caps(self, cap_filepath):
        caps = {}
        with open(cap_filepath, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
            lines = fin.readlines()
        lines = lines[1:]
        for line in lines:
            line = line.strip('\n').strip('\r')
            image_name, caption = line.split(",", 1)
            image_name = image_name.strip().split(".jpg")[0]
            if image_name not in caps:
                caps[image_name] = []
            caps[image_name].append(caption)
        return caps

    def extract_clip_features(self, image_tensor):
        """
        Extract features from the image using the Victim model.

        Args:
            image_tensor (torch.Tensor): The preprocessed image tensor.

        Returns:
            torch.Tensor: The image features extracted by CLIP.
        """
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

    def process_image_and_caption_with_idx(self, idx):
        image_name = self.image_names[idx]
        image_idx = image_name.split(".jpg")[0]
        image_path = os.path.join(self.data_dir, image_name)
        image_tensor = self.process_one_image(image_path)
        image_features = self.extract_clip_features(image_tensor)
        # caption = random.choice(self.captions[image_name.split(".jpg")[0]])
        caption = self.captions[image_idx]
        return image_idx, image_tensor, image_features, caption

    def process_image_and_caption_with_real_image_idx(self, image_idx):
        image_path = os.path.join(self.data_dir, "{}.jpg".format(image_idx))
        image_tensor = self.process_one_image(image_path)
        image_features = self.extract_clip_features(image_tensor)
        caption = self.captions[image_idx]
        return image_idx, image_tensor, image_features, caption

    def get_data_loader(
        self, batch_size = 32, max_samples = None, shuffle = True, collate_fn = None,
    ):
        """
        Get a data loader for the dataset.

        Args:
            batch_size (int, optional): The batch size for the data loader. Defaults to 32.
            max_samples (int, optional): The maximum number of samples to load. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            collate_fn (Callable, optional): The collate function to use. Defaults to None.

        Returns:
            torch.utils.data.DataLoader: The data loader for the dataset.
        """
        from torch.utils.data import DataLoader

        class Filckr8kDataset(Dataset):
            def __init__(
                self, 
                dst_processor, 
                shuffle = True, 
                max_samples = None,
            ):
                self.dst_processor = dst_processor
                self.image_names = dst_processor.image_names
                self.image_ids = list(range(len(self.image_names)))
                if shuffle:
                    random.shuffle(self.image_names)
                if max_samples is not None and max_samples > 0:
                    self.image_names = self.image_names[:max_samples]
                    self.image_ids = self.image_ids[:max_samples]

            def __len__(self):
                return len(self.image_names)

            def __getitem__(self, idx):
                image_name = self.image_names[idx]
                image_idx = image_name.split(".jpg")[0]
                image_idx, image_tensor, image_features, caption = self.dst_processor.process_image_and_caption_with_real_image_idx(image_idx)

                return image_idx, image_tensor, image_features, caption

        dst = Filckr8kDataset(self, shuffle = shuffle, max_samples = max_samples)

        return DataLoader(dst, batch_size = batch_size, collate_fn = collate_fn)


def test_main():
    print("Testing Flickr8kProcessor")
    root_dir = "/root/autodl-tmp/datasets/image_caption_generation/flickr8k/Images"
    processor = Flickr8kProcessor(root_dir)
    print("Number of images: ", len(processor.image_names))
    print("Number of captions: ", len(processor.captions))
    print("Image names: ", processor.image_names[:10])

    data_loader = processor.get_data_loader(batch_size = 2, max_samples = 8000)

    for batch in data_loader:
        print(batch)
        break


def split_flickr8k_and_save():
    """
        Split the Flickr 8k dataset into train, val and test subsets.
    """
    train_num = 6000
    val_num = 1000
    test_num = 1000
    
    root_dir = "/root/autodl-tmp/datasets/image_caption_generation/flickr8k"
    train_dir = os.path.join(root_dir, "flickr8k_train")
    val_dir = os.path.join(root_dir, "flickr8k_val")
    test_dir = os.path.join(root_dir, "flickr8k_test")
    
    image_dir = os.path.join(root_dir, "Images")
    image_files = os.listdir(image_dir)
    random.shuffle(image_files)
    
    train_files = image_files[:train_num]
    val_files = image_files[train_num:train_num + val_num]
    test_files = image_files[train_num + val_num:train_num + val_num + test_num]
    
    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(val_dir, exist_ok = True)
    os.makedirs(test_dir, exist_ok = True)
    
    for image_file in train_files:
        os.system("cp {} {}".format(os.path.join(image_dir, image_file), os.path.join(train_dir, image_file)))
    for image_file in val_files:
        os.system("cp {} {}".format(os.path.join(image_dir, image_file), os.path.join(val_dir, image_file)))
    for image_file in test_files:
        os.system("cp {} {}".format(os.path.join(image_dir, image_file), os.path.join(test_dir, image_file)))


if __name__ == "__main__":
    # test_main()
    split_flickr8k_and_save()
