#!/bin/bash

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model ViT-16B --leaked-feature-layer vit-base

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model ViT-16B --leaked-feature-layer vit-no-proj

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model ViT-32B --leaked-feature-layer vit-base

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model ViT-32B --leaked-feature-layer vit-no-proj

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model RN50 --leaked-feature-layer resnet-base

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model RN50 --leaked-feature-layer resnet-layer1

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model RN50 --leaked-feature-layer resnet-layer2

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model RN50 --leaked-feature-layer resnet-layer3

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model RN50 --leaked-feature-layer resnet-layer4

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model RN101 --leaked-feature-layer resnet-base

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model RN101 --leaked-feature-layer resnet-layer1

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model RN101 --leaked-feature-layer resnet-layer2

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model RN101 --leaked-feature-layer resnet-layer3

python compute_cosine_similarity_with_bailian_text_embedding.py --clip-model RN101 --leaked-feature-layer resnet-layer4