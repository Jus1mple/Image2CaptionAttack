#!/bin/bash

# Train
# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model ViT-32B --leaked-feature-layer vit-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model ViT-16B --leaked-feature-layer vit-base --blip-model blip2-opt-2.7b


python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model ViT-32B --leaked-feature-layer vit-no-proj --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model ViT-16B --leaked-feature-layer vit-no-proj --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN50 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --train-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/train --train-annotation-file None --train-cap-file None --mode train --max-samples -1 --batch-size 4 --num-epochs 6 --victim-model RN101 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b


# Test
# python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model mobilenetv2 --leaked-feature-layer mobilenet-base --blip-model blip2-opt-2.7b


# python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model ViT-32B --leaked-feature-layer vit-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model ViT-16B --leaked-feature-layer vit-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model ViT-32B --leaked-feature-layer vit-no-proj --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model ViT-16B --leaked-feature-layer vit-no-proj --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN50 --leaked-feature-layer resnet-base --blip-model blip2-opt-2.7b

# python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN101 --leaked-feature-layer resnet-base --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN50 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN101 --leaked-feature-layer resnet-layer1 --blip-model blip2-opt-2.7b


python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN50 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN101 --leaked-feature-layer resnet-layer2 --blip-model blip2-opt-2.7b


python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN50 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN101 --leaked-feature-layer resnet-layer3 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN50 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b

python src/main.py --dataset-name imagenet --val-dir /root/autodl-tmp/datasets/image_caption_generation/benjamin-paine/imagenet-1k-256x256/test --val-annotation-file None --val-cap-file None --mode test --max-samples -1 --eval-batch-size 2 --victim-model RN101 --leaked-feature-layer resnet-layer4 --blip-model blip2-opt-2.7b