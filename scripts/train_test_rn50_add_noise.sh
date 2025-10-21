#!/bin/bash


# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer1 --add-noise True

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer1 --add-noise True

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer2 --add-noise True

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer2 --add-noise True

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer3 --add-noise True

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer3 --add-noise True

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer4 --add-noise True

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer4 --add-noise True



python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b --add-noise True

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN50 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b --add-noise True

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b --add-noise True

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN50 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b --add-noise True


python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b --add-noise True

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN50 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b --add-noise True


python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b --add-noise True

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN50 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b --add-noise True


# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b  --add-noise True

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b --add-noise True

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b  --add-noise True

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b --add-noise True

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b  --add-noise True

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b --add-noise True

# python src/main.py --dataset-name flickr8k --train-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_train --train-annotation-file None --train-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode train --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b  --add-noise True

# python src/main.py --dataset-name flickr8k --val-dir /root/autodl-tmp/datasets/image_caption_generation/flickr8k/flickr8k_test --val-annotation-file None --val-cap-file /root/autodl-tmp/datasets/image_caption_generation/flickr8k/captions.txt --mode test --max-samples -1 --batch-size 8 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b --add-noise True