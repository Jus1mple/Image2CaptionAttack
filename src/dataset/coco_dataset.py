# -*- coding:utf-8 -*-
# COCO Dataset


import os
import cv2
import torch
import numpy as np

# from transformers import CLIPProcessor, CLIPModel
from pycocotools.coco import COCO
from tqdm import tqdm
import clip
import random

torch.backends.cudnn.enabled = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.multiprocessing.set_start_method('spawn')

from utils import load_victim_model
from NoiseResNet import NoiseResNetCLIP

class COCOProcessor:
    def __init__(
        self,
        data_dir,
        annotation_file,
        caption_file,
        target_size=224,
        normalize=True,
        clamp_output=False,
        victim_model="ViT-B/32",
        victim_device="cuda:1",
        extract_feature_func=None,
        add_noise = False,
    ):
        """
        Initialize the COCOProcessor class for processing COCO dataset images and captions.

        Args:
            data_dir (str): Directory where the COCO images are stored.
            annotation_file (str): Path to the COCO annotations file (e.g., instances_train2017.json).
            target_size (int): Target size to which the image should be resized.
        """
        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.caption_file = caption_file
        self.target_size = target_size
        self.normalize = normalize
        self.clamp_output = clamp_output
        # self.clip_model = clip_model
        self.victim_device = victim_device
        self.add_noise = add_noise

        self.img_list = os.listdir(data_dir)
        self.img_list = [
            int(img.split(".jpg")[0]) for img in self.img_list if img.endswith(".jpg")
        ]
        self.img_list = sorted(
            self.img_list, reverse=True
        )  # Sort images in descending order
        # self.img_paths = [os.path.join(data_dir, img + ".jpg") for img in self.img_list]

        # Initialize COCO APIs to load coco_anns and coco_caps
        self.coco_anns = COCO(annotation_file)
        self.coco_caps = COCO(caption_file)

        # Set the target size for image resizing
        self.target_size = target_size

        # Initialize CLIP model and processor
        # self.clip_model, _ = clip.load(clip_model, device=clip_device)
        # self.clip_model = self.clip_model.to(torch.float)

        self.victim_model = load_victim_model(victim_model, victim_device)
        self.victim_model = self.victim_model.to(torch.float)
        if self.add_noise:
            self.victim_model = NoiseResNetCLIP(self.victim_model)
        self.extract_feature_func = extract_feature_func

    def get_real_image_id(self, image_idx):
        return self.img_list[image_idx]

    def preprocess_image(self, image_path):
        """
        Preprocess the image to the required format for CLIP.

        Args:
            image_path (str): Path to the image.
            normalize (bool): Whether to normalize the image.
            clamp_output (bool): If True, ensure the output is within [0.0, 1.0] range after normalization.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image
        image = cv2.resize(image, (self.target_size, self.target_size))

        if self.normalize:
            # Normalize the image
            image = cv2.normalize(
                image,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )

        # Convert the image to a PyTorch tensor
        image_tensor = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)

        # Optionally, clamp the image to the [0.0, 1.0] range
        if self.clamp_output:
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

        return image_tensor

    def get_caption(self, image_id):
        """
        Retrieve the caption for a given image ID.

        Args:
            image_id (int): The ID of the image.

        Returns:
            List[str]: The captions of the image.
        """
        image_id = int(str(image_id).zfill(12))
        # Get all annotations for this image
        ann_ids = self.coco_caps.getAnnIds(imgIds=image_id)
        anns = self.coco_caps.loadAnns(ann_ids)
        return [ann["caption"] for ann in anns]

    def process_image_and_caption_with_idx(self, image_idx):
        """
        Process an image and its caption from the COCO dataset.

        Args:
            image_idx (int): The IDX of the image in image list.

        Returns:
            tuple: (image id, image tensor, image features, caption)
        """
        # Load the image and preprocess
        # img_info = self.coco.loadImgs([image_id])[0]
        # image_path = os.path.join(self.data_dir, img_info['file_name'])
        image_id = str(self.get_real_image_id(image_idx)).zfill(12)
        # image_path = self.img_paths[image_idx]
        image_path = os.path.join(self.data_dir, image_id + ".jpg")

        # Get the image tensor
        image_tensor = self.preprocess_image(image_path)

        # Extract image features using CLIP
        image_features = self.extract_clip_features(image_tensor)

        # Get the caption for the image
        caption = self.get_caption(image_id)

        return image_id, image_tensor, image_features, caption

    def process_image_and_caption_with_real_id(self, image_id):
        """
        Process an image and its caption from the COCO dataset.

        Args:
            image_id (int): The ID of the image in COCO dataset.

        Returns:
            tuple: (image id, image tensor, image features, caption)
        """
        # Load the image and preprocess
        # img_info = self.coco.loadImgs([image_id])[0]
        # image_path = os.path.join(self.data_dir, img_info['file_name'])

        # image_id = "000000000000"
        # image_id = image_id[:12-len(str(image_id))] + str(image_id)
        # print(image_id)
        image_id = str(image_id).zfill(12)
        image_path = os.path.join(self.data_dir, image_id + ".jpg")

        # Get the image tensor
        image_tensor = self.preprocess_image(image_path)

        # Extract image features using CLIP
        image_features = self.extract_clip_features(image_tensor)

        # Get the caption for the image
        caption = self.get_caption(image_id)

        return image_id, image_tensor, image_features, caption

    def extract_clip_features(self, image_tensor):
        """
        Extract features from the image using the CLIP model.

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

    def get_data_loader(
        self, batch_size=32, max_samples=None, shuffle=True, collate_fn=None
    ):
        """
        Generate a data loader for the COCO dataset to fetch images and captions.

        Args:
            batch_size (int): The batch size for the data loader.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the COCO dataset.
        """
        from torch.utils.data import Dataset, DataLoader

        class COCODataLoader(Dataset):
            def __init__(self, coco_processor, shuffle=False, max_samples=None):
                self.coco_processor = coco_processor
                self.image_ids = list(range(len(coco_processor.img_list)))
                if shuffle:
                    random.shuffle(self.image_ids)
                self.image_real_ids = [
                    coco_processor.get_real_image_id(image_idx)
                    for image_idx in self.image_ids
                ]
                if max_samples is not None and max_samples > 0:
                    self.image_ids = self.image_ids[:max_samples]
                    self.image_real_ids = self.image_real_ids[:max_samples]

            def __len__(self):
                return len(self.image_ids)

            def __getitem__(self, idx):
                image_idx = self.image_ids[idx]
                image_id, image_tensor, image_features, caption = (
                    self.coco_processor.process_image_and_caption_with_idx(
                        image_idx=image_idx
                    )
                )
                return image_id, image_tensor, image_features, caption

        coco_dataset = COCODataLoader(
            self, shuffle=shuffle, max_samples=max_samples
        )  # Adjust max_samples as needed
        return DataLoader(coco_dataset, batch_size=batch_size, collate_fn=collate_fn)

    def __len__(self):
        return len(self.img_list)


def main():
    # Example Usage:
    print("Processing COCO dataset...")
    dataDir = "/hub/huggingface/datasets/COCO"
    dataType = "train2017"
    train_dir = os.path.join(dataDir, dataType)
    train_annotation_file = "{}/annotations/instances_{}.json".format(dataDir, dataType)
    capFile = "{}/annotations/captions_{}.json".format(dataDir, dataType)
    print("Creating COCOProcessor...")
    coco_processor = COCOProcessor(train_dir, train_annotation_file, capFile)

    print("Processing a single image and caption...")
    # Process a single image and caption
    image_id = 318099  # Example image ID from COCO
    image_tensor, image_features, caption = (
        coco_processor.process_image_and_caption_with_real_id(image_id)
    )

    print("Image Tensor:", image_tensor.shape)
    print("Image Features:", image_features.shape)
    print("Caption:", caption)

    print("Creating DataLoader...")
    # Get a DataLoader to fetch batches of data
    data_loader = coco_processor.get_data_loader(
        batch_size=8, max_samples=32, shuffle=True
    )

    print("Iterating over DataLoader batches...")
    # Iterate over batches
    for batch in tqdm(data_loader, desc="Processing batches"):
        image_tensors, image_features, captions = batch
        print(image_tensors.shape, image_features.shape, len(captions))
        print(captions)
        break


if __name__ == "__main__":
    main()
