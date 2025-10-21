#!/bin/bash


# COCO2017

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv2 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-base

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv2 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer1

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv2 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer-mid

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv2 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-all-blocks


python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv3-small --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-base

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv3-small --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer1

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv3-small --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer-mid

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv3-small --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-all-blocks


python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv3-large --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-base

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv3-large --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer1

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv3-large --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-layer-mid

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model mobilenetv3-large --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer mobilenet-all-blocks


# Flickr8k

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b



python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b


python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b

python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b


# ImageNet

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b

python src/main.py --dataset-na
me imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b


python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b


python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-layer1 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-layer-mid --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-all-blocks --blip-model blip2-opt-2.7b