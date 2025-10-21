#!/bin/bash


# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b 

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b 

# COCO Dataset
# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv2 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-base --batch-size 16

python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv2 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer1 --batch-size 16

python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv2 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-all-blocks --batch-size 16

python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv2 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer-mid --batch-size 16


# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv3-small --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-base

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv3-small --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer1

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv3-small --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-all-blocks

python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv3-small --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer-mid


# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv3-large --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-base

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv3-large --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer1

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv3-large --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-all-blocks

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model mobilenetv3-large --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer-mid

# # Flickr8k Dataset
# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b 

python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b 

python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b 

python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b 


# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b 

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b 

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b 

python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b 


# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b 

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b 

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b 

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b 


# # ImageNet Dataset
# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b


# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b


# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b