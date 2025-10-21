#!/bin/bash


# # python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model ViT-16B --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer vit-base

# # python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model ViT-16B --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer vit-no-proj

# # python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model ViT-32B --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer vit-base

# # python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model ViT-32B --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer vit-no-proj

# # python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-base

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer1

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer2

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer3

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer4

# # python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN101 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-base

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN101 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer1

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN101 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer2

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN101 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer3

# python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN101 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer4

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer1

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer2

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer3

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer4

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN101 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer1

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN101 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer2

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN101 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer3

# python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN101 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer4



python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer1 --add-noise True

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer1 --add-noise True

python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer2 --add-noise True

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer2 --add-noise True

python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer3 --add-noise True

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer3 --add-noise True

python src/main.py --mode train --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer4 --add-noise True

python src/main.py --mode test --blip-model blip2-opt-2.7b --victim-model RN50 --model-device cuda:0 --victim-device cuda:0 --leaked-feature-layer resnet-layer4 --add-noise True