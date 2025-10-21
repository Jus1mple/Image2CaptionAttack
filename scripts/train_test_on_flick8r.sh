#!/bin/bash

# TRAIN

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model ViT-32B --leaked-feature-layer vit-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model ViT-32B --leaked-feature-layer vit-no-proj --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model ViT-16B --leaked-feature-layer vit-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model ViT-16B --leaked-feature-layer vit-no-proj --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b


# TEST

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model ViT-32B --leaked-feature-layer vit-base --blip-model blip2-opt-2.7b


# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model ViT-32B --leaked-feature-layer vit-no-proj --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model ViT-16B --leaked-feature-layer vit-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model ViT-16B --leaked-feature-layer vit-no-proj --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b