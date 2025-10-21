#!/bin/bash


python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model ViT-16B --leaked-feature-layer vit-base

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model ViT-16B --leaked-feature-layer vit-no-proj

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model ViT-32B --leaked-feature-layer vit-base

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model ViT-32B --leaked-feature-layer vit-no-proj

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model RN50 --leaked-feature-layer resnet-base

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model RN50 --leaked-feature-layer resnet-layer1

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model RN50 --leaked-feature-layer resnet-layer2

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model RN50 --leaked-feature-layer resnet-layer3

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model RN50 --leaked-feature-layer resnet-layer4

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model RN101 --leaked-feature-layer resnet-base

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model RN101 --leaked-feature-layer resnet-layer1

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model RN101 --leaked-feature-layer resnet-layer2

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model RN101 --leaked-feature-layer resnet-layer3

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model RN101 --leaked-feature-layer resnet-layer4

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-base

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-layer1

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-layer-mid

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv3-large --leaked-feature-layer mobilenet-all-blocks

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-base

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer1

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-layer-mid

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv3-small --leaked-feature-layer mobilenet-all-blocks


python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv2 --leaked-feature-layer mobilenet-base

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv2 --leaked-feature-layer mobilenet-layer1

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv2 --leaked-feature-layer mobilenet-layer-mid

python src/create_imagenet_format_results_for_eval.py --dataset-name imagenet --victim-model mobilenetv2 --leaked-feature-layer mobilenet-all-blocks