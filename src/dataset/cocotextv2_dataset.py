# -*- coding:utf-8 -*-
# COCOText-V2.0 Dataset for Text Recognition


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



class COCOTextProcessor:
    def __init__(
        self, 
        data_dir, 
        annotation_file, 
        caption_file, 
        target_size = 224, 
        normalize=True,
        clamp_output=False,
        victim_model="ViT-B/32",
        victim_device="cuda:1",
        extract_feature_func=None,
    ):
        pass