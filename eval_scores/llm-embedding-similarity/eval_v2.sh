#!/bin/bash


python compute_cosine_similarity_with_bailian_text_embedding.py --dataset-name flickr8k --clip-model ViT-16B --leaked-feature-layer vit-base

python compute_cosine_similarity_with_bailian_text_embedding.py --dataset-name flickr8k --clip-model ViT-32B --leaked-feature-layer vit-base

python compute_cosine_similarity_with_bailian_text_embedding.py --dataset-name flickr8k --clip-model RN50 --leaked-feature-layer resnet-base

python compute_cosine_similarity_with_bailian_text_embedding.py --dataset-name flickr8k --clip-model RN101 --leaked-feature-layer resnet-base

python compute_cosine_similarity_with_bailian_text_embedding.py --dataset-name flickr8k --clip-model mobilenetv2 --leaked-feature-layer mobilenet-base





python compute_cosine_similarity_with_bailian_text_embedding.py --dataset-name imagenet --clip-model ViT-16B --leaked-feature-layer vit-base

python compute_cosine_similarity_with_bailian_text_embedding.py --dataset-name imagenet --clip-model ViT-32B --leaked-feature-layer vit-base

python compute_cosine_similarity_with_bailian_text_embedding.py --dataset-name imagenet --clip-model RN50 --leaked-feature-layer resnet-base

python compute_cosine_similarity_with_bailian_text_embedding.py --dataset-name imagenet --clip-model RN101 --leaked-feature-layer resnet-base

python compute_cosine_similarity_with_bailian_text_embedding.py --dataset-name imagenet --clip-model mobilenetv2 --leaked-feature-layer mobilenet-base
