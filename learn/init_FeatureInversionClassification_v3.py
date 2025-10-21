import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F
# from transformers import CLIPProcessor, CLIPModel, BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer
import clip
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader, Subset, Dataset
import os
import random
from typing import Dict, List, Tuple, Optional, Union

class DatasetLoader:
    """Handles loading and preprocessing of various image classification datasets"""
    
    SUPPORTED_DATASETS = {
        'cifar10': {
            'num_classes': 10,
            'classes': [
                'airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck'
            ],
            'img_size': 32,
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2470, 0.2435, 0.2616]
        },
        'cifar100': {
            'num_classes': 100,
            'img_size': 32,
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761]
        },
        'imagenet': {
            'num_classes': 1000,
            'img_size': 224,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'mnist': {
            'num_classes': 10,
            'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'img_size': 28,
            'mean': [0.1307],
            'std': [0.3081]
        },
        'fashion_mnist': {
            'num_classes': 10,
            'classes': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
            'img_size': 28,
            'mean': [0.2860],
            'std': [0.3530]
        }
    }

    @staticmethod
    def get_dataset_info(dataset_name: str):
        """Get dataset information"""
        if dataset_name not in DatasetLoader.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} is not supported. Supported datasets: {list(DatasetLoader.SUPPORTED_DATASETS.keys())}")
        return DatasetLoader.SUPPORTED_DATASETS[dataset_name]
    
    @staticmethod
    def get_class_names(dataset_name: str) -> List[str]:
        """Get list of class names for the dataset"""
        info = DatasetLoader.get_dataset_info(dataset_name)
        if 'classes' in info:
            return info['classes']
        elif dataset_name == 'imagenet':
            # Load ImageNet class names from file
            try:
                import json
                with open('imagenet_classes.json', 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                print("ImageNet class names file not found. Creating a default list...")
                # Return generic class names if file not found
                return [f"class_{i}" for i in range(1000)]
        elif dataset_name == 'cifar100':
            # Load CIFAR-100 class names
            try:
                dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
                return dataset.classes
            except:
                return [f"class_{i}" for i in range(100)]
        else:
            return [f"class_{i}" for i in range(info['num_classes'])]
    
    @staticmethod
    def get_transform(dataset_name: str, for_clip: bool = False) -> transforms.Compose:
        """Get appropriate transforms for the dataset"""
        info = DatasetLoader.get_dataset_info(dataset_name)
        
        if for_clip:
            # CLIP needs 224x224 inputs
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224) if dataset_name == 'imagenet' else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                     std=[0.26862954, 0.26130258, 0.27577711])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(info['img_size']),
                transforms.CenterCrop(info['img_size']) if dataset_name == 'imagenet' else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=info['mean'], std=info['std'])
            ])
    
    @staticmethod
    def load_dataset(dataset_name: str, batch_size: int = 64, sample_size: int = 1000, data_dir: str = './data') -> Tuple[DataLoader, List[str]]:
        """Load and prepare the specified dataset"""
        info = DatasetLoader.get_dataset_info(dataset_name)
        transform = DatasetLoader.get_transform(dataset_name)
        
        print(f"Loading {dataset_name} dataset...")
        
        # Load the appropriate dataset
        if dataset_name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir, dataset_name), train=False, download=True, transform=transform)
        elif dataset_name == 'cifar100':
            dataset = torchvision.datasets.CIFAR100(root=os.path.join(data_dir, dataset_name), train=False, download=True, transform=transform)
        elif dataset_name == 'mnist':
            dataset = torchvision.datasets.MNIST(root=os.path.join(data_dir, dataset_name), train=False, download=True, transform=transform)
        elif dataset_name == 'fashion_mnist':
            dataset = torchvision.datasets.FashionMNIST(root=os.path.join(data_dir, dataset_name), train=False, download=True, transform=transform)
        elif dataset_name == 'imagenet':
            # For ImageNet, we need to check if the dataset is available
            try:
                # Attempt to load the ImageNet validation set
                dataset = torchvision.datasets.ImageNet(
                    root=os.path.join(data_dir, dataset_name),
                    split='val',
                    transform=transform
                )
                print("Successfully loaded ImageNet dataset!")
            except (RuntimeError, FileNotFoundError):
                print("Full ImageNet dataset not found. Loading ImageNet subset or using ImageNetV2...")
                try:
                    # Try loading ImageNetV2 (smaller test set)
                    from imagenetv2_pytorch import ImageNetV2Dataset
                    dataset = ImageNetV2Dataset(transform=transform)
                except ImportError:
                    raise ImportError(
                        "ImageNet dataset not found and ImageNetV2 package not installed. "
                        "Please either download the ImageNet dataset or install ImageNetV2: "
                        "pip install imagenetv2-pytorch"
                    )
        else:
            raise ValueError(f"Dataset {dataset_name} loading not implemented")
        
        # Get class names
        class_names = DatasetLoader.get_class_names(dataset_name)
        
        # Subsample if needed
        if sample_size and sample_size < len(dataset):
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            subset_indices = indices[:sample_size]
            dataset = Subset(dataset, subset_indices)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        return dataloader, class_names


class FeatureInversionAttack:
    """
    Implements feature inversion attack that steals and reverses intermediate 
    layer features using PGD algorithm and feature-text alignment.
    Modified to work with different datasets.
    """
    def __init__(self, dataset_name: str = 'cifar10', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.dataset_name = dataset_name

        # Get dataset information
        self.dataset_info = DatasetLoader.get_dataset_info(dataset_name)
        self.class_names = DatasetLoader.get_class_names(dataset_name)
        self.num_classes = self.dataset_info['num_classes']

        # Initialize model components
        print("Loading CLIP model...")
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.clip_model, self.processor = clip.load("ViT-B/32", device=device)
        
        # Freeze original CLIP model parameters
        # for param in self.clip_model.parameters():
        #     param.requires_grad = False
        
        self.clip_model = self.clip_model.to(torch.float32)
        
        # Initialize Q-Former for feature-text alignment
        self.q_former = self._init_q_former().to(device)

        # Transform for processing images for CLIP
        self.transform = DatasetLoader.get_transform(dataset_name, for_clip=True)

    def _init_q_former(self):
        """Initialize Q-Former module for feature-text alignment"""
        # Use pre-trained BERT as the base for Q-Former
        bert_model = BertModel.from_pretrained("bert-base-uncased")

        # Define Q-Former architecture
        class QFormer(nn.Module):
            def __init__(self, bert_model, img_feature_dim=512, text_feature_dim=768, num_classes=10):
                super(QFormer, self).__init__()
                self.bert = bert_model
                self.img_projection = nn.Linear(img_feature_dim, text_feature_dim)
                self.classifier = nn.Linear(text_feature_dim, num_classes)

            def forward(self, img_features, text=None):
                # Project image features to text space
                projected_img_features = self.img_projection(img_features)

                if text is not None:
                    # Extract text features
                    outputs = self.bert(**text)
                    text_features = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token

                    # Calculate similarity between image and text features
                    similarity = F.cosine_similarity(projected_img_features, text_features, dim=1)
                    return similarity, self.classifier(projected_img_features)
                else:
                    # Classification only
                    return self.classifier(projected_img_features)

        return QFormer(bert_model, img_feature_dim=512, num_classes=self.num_classes)

    def load_dataset(self, batch_size=64, sample_size=1000, data_dir='./data'):
        """Load and preprocess the selected dataset"""
        return DatasetLoader.load_dataset(self.dataset_name, batch_size, sample_size, data_dir)

    def steal_features(self, image):
        """Steal intermediate layer features from CLIP model"""
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL image
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # Convert PyTorch tensor to PIL image
            if image.dim() == 4:  # batch
                image = image[0]  # take first image
            image = transforms.ToPILImage()(image)

        # Ensure image is RGB format
        image = image.convert("RGB")

        # Process image
        # inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        inputs = self.processor(image).unsqueeze(0).to(self.device)

        # Get image features from CLIP model
        with torch.no_grad():
            # image_features = self.clip_model.get_image_features(**inputs)
            image_features = self.clip_model.encode_image(inputs)

        return image_features

    def steal_features_batch(self, images, labels):
        """Batch steal features from multiple images"""
        features_list = []
        labels_list = []

        for img, label in zip(images, labels):
            # Convert tensor to PIL image
            if len(img.shape) == 3:  # RGB image
                img_pil = transforms.ToPILImage()(img)
            else:  # Grayscale image (e.g., MNIST)
                img_pil = transforms.ToPILImage()(img.repeat(3, 1, 1))  # Convert to RGB

            # Steal features
            feature = self.steal_features(img_pil)
            features_list.append(feature)
            labels_list.append(label)

        # Stack features and labels into batches
        features_batch = torch.cat(features_list, dim=0)
        labels_batch = torch.tensor(labels_list).to(self.device)

        return features_batch, labels_batch

    def train_q_former(self, dataloader, epochs=10):
        """
        Train Q-Former module on the dataset
        
        Parameters:
        - dataloader: Dataset loader
        - epochs: Number of training epochs
        """
        print(f"Preparing {self.dataset_name} text descriptions...")
        # Prepare text descriptions for classes
        text_descriptions = [f"a photo of a {cls}" for cls in self.class_names]

        # Process text descriptions
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_inputs = tokenizer(text_descriptions, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Set optimizer
        optimizer = optim.Adam(self.q_former.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        print(f"Starting Q-Former training for {epochs} epochs...")
        # Training loop
        for epoch in range(epochs):
            self.q_former.train()
            epoch_loss = 0.0
            batch_count = 0

            for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move images and labels to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Handle grayscale images (MNIST, FashionMNIST)
                if images.shape[1] == 1:  # If grayscale (1 channel)
                    images = images.repeat(1, 3, 1, 1)  # Convert to RGB

                # Steal features
                stolen_features, _ = self.steal_features_batch(images, labels)

                # Reset optimizer gradients
                optimizer.zero_grad()

                # Create text inputs for corresponding labels
                batch_text_inputs = {
                    k: torch.cat([text_inputs[k][labels[i]].unsqueeze(0) for i in range(len(labels))], dim=0)
                    for k in text_inputs.keys()
                }

                # Forward pass
                similarity, logits = self.q_former(stolen_features, batch_text_inputs)

                # Calculate loss: classification loss + contrastive loss
                classification_loss = criterion(logits, labels)
                contrastive_loss = -torch.mean(similarity)  # Maximize similarity

                loss = classification_loss + 0.5 * contrastive_loss

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_epoch_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")

            # Evaluate model after each epoch
            self.evaluate_q_former(dataloader)

    def evaluate_q_former(self, dataloader):
        """Evaluate Q-Former performance on the dataset"""
        self.q_former.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                # Move images and labels to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Handle grayscale images
                if images.shape[1] == 1:  # If grayscale (1 channel)
                    images = images.repeat(1, 3, 1, 1)  # Convert to RGB

                # Steal features
                stolen_features, _ = self.steal_features_batch(images, labels)

                # Predict classes
                logits = self.q_former(stolen_features)
                predictions = torch.argmax(logits, dim=1)

                # Count correct predictions
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Q-Former Accuracy on {self.dataset_name}: {accuracy:.2f}%")
        return accuracy

    def pgd_attack(
        self,
        target_features,
        target_class,
        epsilon=0.1,
        alpha=0.01,
        steps=100,
        reconstruct_image=True,
    ):
        """
        Use PGD algorithm to invert stolen features and generate a possible image representation

        Parameters:
        - target_features: Target (stolen) features
        - target_class: Target class
        - epsilon: Maximum range of PGD perturbation
        - alpha: PGD step size
        - steps: Number of PGD iterations
        - reconstruct_image: Whether to attempt image reconstruction

        Returns:
        - inverted_features: Optimized features
        - reconstructed_image: Reconstructed image (if reconstruct_image=True)
        """
        if target_class >= len(self.class_names):
            raise ValueError(
                f"Target class {target_class} out of range (0-{len(self.class_names)-1})"
            )

        print(f"Starting PGD attack for class '{self.class_names[target_class]}'...")

        # Initialize a random image feature as starting point
        x = torch.randn_like(target_features).requires_grad_(True)

        # Prepare target class text description
        text_description = f"a photo of a {self.class_names[target_class]}"

        # Process text description
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_input = tokenizer(
            [text_description], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # Set optimizer
        optimizer = optim.Adam([x], lr=alpha)

        # Original feature range
        x_orig = x.clone().detach()

        # Save loss history for plotting
        loss_history = []

        # PGD attack loop
        for i in tqdm(range(steps)):
            # Reset gradients
            optimizer.zero_grad()

            # Calculate feature distance loss
            feature_loss = F.mse_loss(x, target_features)

            # Calculate classification loss
            logits = self.q_former(x)
            target = torch.tensor([target_class]).to(self.device)
            classification_loss = F.cross_entropy(logits, target)

            # Calculate total loss
            loss = feature_loss + classification_loss
            loss_history.append(loss.item())

            # Backward propagation
            loss.backward()

            # Update step
            optimizer.step()

            # Project back to ε-ball
            with torch.no_grad():
                delta = x - x_orig
                delta = torch.clamp(delta, -epsilon, epsilon)
                x.data = x_orig + delta

            if (i + 1) % 20 == 0:
                print(f"Step {i+1}/{steps}, Loss: {loss.item():.4f}")

        # Plot loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history)
        plt.title("PGD Attack Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(f"pgd_loss_class_{target_class}.png")

        inverted_features = x.detach()

        # Image reconstruction using CLIP's image decoder capabilities
        if reconstruct_image:
            reconstructed_image = self.reconstruct_image_from_features(inverted_features)
            return inverted_features, reconstructed_image

        return inverted_features

    def reconstruct_image_from_features(self, features):
        """
        Reconstruct an image from optimized CLIP image features

        Parameters:
        - features: Optimized image features

        Returns:
        - reconstructed_image: Reconstructed PIL image
        """
        print("Reconstructing image from features...")

        # Define image size based on dataset
        if self.dataset_name in ["mnist", "fashion_mnist"]:
            img_size = 28
            channels = 1
        else:
            img_size = self.dataset_info["img_size"]
            channels = 3

        # Initialize a random image
        reconstructed_tensor = torch.randn(1, channels, img_size, img_size).to(self.device)
        reconstructed_tensor.requires_grad_(True)

        # Set optimizer for image reconstruction
        rec_optimizer = optim.Adam([reconstructed_tensor], lr=0.01)

        # Number of steps for image reconstruction
        rec_steps = 300

        # Save reconstruction loss history
        rec_loss_history = []

        # Normalize input tensor
        def normalize_tensor(t):
            return (t - t.min()) / (t.max() - t.min() + 1e-5)

        # Reconstruction loop
        for i in tqdm(range(rec_steps), desc="Image Reconstruction"):
            # Reset gradients
            rec_optimizer.zero_grad()

            # Get image features for current tensor
            if channels == 1:  # Handle grayscale images
                img_tensor = reconstructed_tensor.repeat(
                    1, 3, 1, 1
                )  # Convert to RGB for CLIP
            else:
                img_tensor = reconstructed_tensor

            # Resize if needed
            if img_size != 224:
                img_tensor = F.interpolate(
                    img_tensor, size=224, mode="bilinear", align_corners=False
                )

            # Normalize for CLIP
            img_tensor = normalize_tensor(img_tensor)

            # Get features through CLIP model
            current_features = self._get_features_from_image_tensor(img_tensor)

            # print("Current features shape:", current_features.shape)
            # print("Target features shape:", features.shape)
            
            # Calculate feature reconstruction loss
            # loss = F.mse_loss(current_features, features)
            loss = F.pairwise_distance(current_features, features, p=2).mean()
            
            # print("Reconstruction loss:", loss.item())
            
            rec_loss_history.append(loss.item())

            # Backward pass
            loss.backward()
            rec_optimizer.step()

            # Optional: Add TV loss or other regularization

            if (i + 1) % 50 == 0:
                print(f"Reconstruction step {i+1}/{rec_steps}, Loss: {loss.item():.4f}")

        # Plot reconstruction loss
        plt.figure(figsize=(10, 5))
        plt.plot(rec_loss_history)
        plt.title("Image Reconstruction Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(f"reconstruction_loss.png")

        # Convert final tensor to PIL image
        with torch.no_grad():
            final_tensor = reconstructed_tensor.squeeze(0).cpu()
            final_tensor = normalize_tensor(final_tensor) * 255

            if channels == 1:
                reconstructed_image = transforms.ToPILImage()(final_tensor.byte())
            else:
                reconstructed_image = transforms.ToPILImage()(final_tensor.byte())

        return reconstructed_image

    def _get_features_from_image_tensor(self, img_tensor):
        """Helper method to get features from an image tensor"""
        # Convert to PIL image
        with torch.enable_grad():
            # Process the image through CLIP's preprocessor
            # Make a copy to avoid modifying the original tensor
            # inputs = {}
            # inputs["pixel_values"] = img_tensor.detach()

            # Get image features from CLIP model
            # image_features = self.clip_model.get_image_features(**inputs)
            image_features = self.clip_model.encode_image(img_tensor.detach())

        return image_features

    def evaluate_attack(self, inverted_features, ground_truth_class):
        """Evaluate attack effectiveness"""
        self.q_former.eval()
        with torch.no_grad():
            logits = self.q_former(inverted_features)
            probabilities = F.softmax(logits, dim=1)

            # Get probabilities for all classes
            probs_dict = {self.class_names[i]: probabilities[0, i].item() for i in range(min(len(self.class_names), self.num_classes))}

            # Get predicted class
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        success = (predicted_class == ground_truth_class)

        print(f"\nAttack Result:")
        print(f"Target Class: {self.class_names[ground_truth_class]} (ID: {ground_truth_class})")
        print(f"Predicted Class: {self.class_names[predicted_class]} (ID: {predicted_class})")
        print(f"Success: {success}, Confidence: {confidence:.4f}")

        # Show probabilities for all classes (top 10 if many classes)
        print("\nClass Probabilities:")
        top_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        for cls, prob in top_probs:
            print(f"{cls}: {prob:.4f}")

        return success, confidence, probs_dict

    def feature_only_attack(self, stolen_features):
        """
        Second attack method: directly predict class from stolen features
        without reconstructing the original image

        Parameters:
        - stolen_features: Target (stolen) features from victim model

        Returns:
        - predicted_class: Predicted class ID
        - confidence: Confidence score for the prediction
        - all_probs: Probabilities for all classes
        """
        self.q_former.eval()
        with torch.no_grad():
            # Direct classification from features
            logits = self.q_former(stolen_features)
            probabilities = F.softmax(logits, dim=1)

            # Get predicted class
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

            # Get probabilities for all classes
            all_probs = {
                self.class_names[i]: probabilities[0, i].item()
                for i in range(min(len(self.class_names), self.num_classes))
            }

        return predicted_class, confidence, all_probs

    def run_attack_experiment(
        self, 
        num_epochs=5,
        num_samples=5, 
        steps=150, 
        data_dir='./data', 
        epsilon=0.2, 
        alpha=0.01, 
        attack_type='both',
        save_reconstructed_images=True
    ):
        """
        Run complete attack experiment with support for multiple attack types
        
        Parameters:
        - num_epochs: Number of epochs for training Q-Former
        - num_samples: Number of samples to test per class
        - steps: Number of PGD attack steps
        - data_dir: Directory for dataset storage
        - epsilon: Maximum perturbation range
        - alpha: PGD step size
        - attack_type: Type of attack to run ('inversion', 'feature_only', or 'both')
        - save_reconstructed_images: Whether to save reconstructed images
        """
        # Create directories for saving results
        if save_reconstructed_images:
            recon_dir = 'reconstructed_images'
            os.makedirs(recon_dir, exist_ok=True)

        # Load dataset
        dataloader, _ = self.load_dataset(batch_size=32, sample_size=1000, data_dir=data_dir)

        # Train Q-Former
        self.train_q_former(dataloader, epochs=num_epochs)

        # Select classes to test based on dataset size
        if self.num_classes <= 10:
            # Test all classes
            test_classes = list(range(self.num_classes))
        else:
            # If too many classes (e.g., ImageNet), randomly select 10
            test_classes = random.sample(range(self.num_classes), 10)

        # Results storage
        results = {
            'inversion_attack': {},
            'feature_only_attack': {}
        }

        # Collect samples for each selected class
        class_samples = {i: [] for i in test_classes}

        for images, labels in dataloader:
            for img, label in zip(images, labels):
                label = label.item()
                if label in test_classes and len(class_samples[label]) < num_samples:
                    class_samples[label].append(img)

        # Attack samples for each class
        for class_id in test_classes:
            if len(class_samples[class_id]) == 0:
                print(f"No samples found for class {class_id}. Skipping.")
                continue

            class_name = self.class_names[class_id]
            print(f"\n{'='*50}")
            print(f"Testing attack on class: {class_name}")
            print(f"{'='*50}")

            inversion_results = []
            feature_only_results = []

            for i, img in enumerate(class_samples[class_id]):
                print(f"\nSample {i+1}/{len(class_samples[class_id])} for class {class_name}")

                # Handle grayscale images
                if img.shape[0] == 1:  # If grayscale (1 channel)
                    img = img.repeat(3, 1, 1)  # Convert to RGB

                # Convert tensor to PIL image for visualization
                original_pil = transforms.ToPILImage()(img.cpu())

                # Steal features
                target_features = self.steal_features(img)

                # Attack Type 1: Feature Inversion Attack (with PGD)
                if attack_type in ['inversion', 'both']:
                    print("\nRunning Feature Inversion Attack...")
                    # Execute PGD attack with image reconstruction
                    inverted_features, reconstructed_image = self.pgd_attack(
                        target_features, 
                        class_id,
                        epsilon=epsilon, 
                        alpha=alpha, 
                        steps=steps,
                        reconstruct_image=True
                    )

                    # Save reconstructed image
                    if save_reconstructed_images:
                        recon_filename = f"{recon_dir}/{self.dataset_name}_{class_name}_sample{i}.png"
                        reconstructed_image.save(recon_filename)
                        print(f"Saved reconstructed image to {recon_filename}")

                        # Also save original image for comparison
                        orig_filename = f"{recon_dir}/{self.dataset_name}_{class_name}_sample{i}_original.png"
                        original_pil.save(orig_filename)

                        # Create side by side comparison
                        comparison = Image.new('RGB', (reconstructed_image.width*2, reconstructed_image.height))
                        comparison.paste(original_pil.resize((reconstructed_image.width, reconstructed_image.height)), (0, 0))
                        comparison.paste(reconstructed_image, (reconstructed_image.width, 0))
                        comp_filename = f"{recon_dir}/{self.dataset_name}_{class_name}_sample{i}_comparison.png"
                        comparison.save(comp_filename)
                        print(f"Saved comparison image to {comp_filename}")

                    # Calculate reconstruction quality metrics
                    if hasattr(reconstructed_image, 'size'):  # Ensure the reconstruction was successful
                        # Convert PIL to tensor for metrics calculation
                        rec_tensor = transforms.ToTensor()(reconstructed_image).unsqueeze(0).to(self.device)
                        img_tensor = img.unsqueeze(0).to(self.device)

                        # Resize for comparison if sizes differ
                        if rec_tensor.shape[-1] != img_tensor.shape[-1]:
                            rec_tensor = F.interpolate(rec_tensor, size=img_tensor.shape[-2:], mode='bilinear')

                        # PSNR (Peak Signal-to-Noise Ratio)
                        mse = F.mse_loss(rec_tensor, img_tensor).item()
                        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0

                        # SSIM could be calculated with skimage if available
                        recon_quality = {
                            'psnr': psnr,
                            'mse': mse
                        }
                    else:
                        recon_quality = {'error': 'Reconstruction failed'}

                    # Evaluate attack effectiveness
                    success, confidence, probs = self.evaluate_attack(inverted_features, class_id)

                    inversion_results.append({
                        'success': success,
                        'confidence': confidence,
                        'probabilities': probs,
                        'reconstruction_quality': recon_quality
                    })

                # Attack Type 2: Feature-Only Attack
                if attack_type in ['feature_only', 'both']:
                    print("\nRunning Feature-Only Attack...")
                    predicted_class, confidence, probs = self.feature_only_attack(target_features)

                    # Determine if prediction is successful
                    success = (predicted_class == class_id)

                    print(f"Feature-Only Attack Result:")
                    print(f"True Class: {self.class_names[class_id]} (ID: {class_id})")
                    print(f"Predicted Class: {self.class_names[predicted_class]} (ID: {predicted_class})")
                    print(f"Success: {success}, Confidence: {confidence:.4f}")

                    # Show top probabilities
                    print("\nTop Class Probabilities:")
                    top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                    for cls, prob in top_probs:
                        print(f"{cls}: {prob:.4f}")

                    feature_only_results.append({
                        'success': success,
                        'confidence': confidence,
                        'probabilities': probs
                    })

            # Summarize results for this class
            if attack_type in ['inversion', 'both'] and inversion_results:
                inv_success_rate = sum(res['success'] for res in inversion_results) / len(inversion_results)
                inv_avg_confidence = sum(res['confidence'] for res in inversion_results) / len(inversion_results)

                # Average reconstruction quality metrics
                avg_recon_quality = {}
                for metric in ['psnr', 'mse']:
                    values = [res['reconstruction_quality'].get(metric) for res in inversion_results 
                            if metric in res.get('reconstruction_quality', {})]
                    if values:
                        avg_recon_quality[metric] = sum(values) / len(values)

                results['inversion_attack'][class_name] = {
                    'success_rate': inv_success_rate,
                    'avg_confidence': inv_avg_confidence,
                    'avg_reconstruction_quality': avg_recon_quality,
                    'detailed_results': inversion_results
                }

                print(f"\nClass {class_name} - Inversion Attack Results:")
                print(f"Success Rate: {inv_success_rate*100:.2f}%")
                print(f"Average Confidence: {inv_avg_confidence:.4f}")
                if 'psnr' in avg_recon_quality:
                    print(f"Average PSNR: {avg_recon_quality['psnr']:.2f} dB")
                if 'mse' in avg_recon_quality:
                    print(f"Average MSE: {avg_recon_quality['mse']:.4f}")

            if attack_type in ['feature_only', 'both'] and feature_only_results:
                feat_success_rate = sum(res['success'] for res in feature_only_results) / len(feature_only_results)
                feat_avg_confidence = sum(res['confidence'] for res in feature_only_results) / len(feature_only_results)

                results['feature_only_attack'][class_name] = {
                    'success_rate': feat_success_rate,
                    'avg_confidence': feat_avg_confidence,
                    'detailed_results': feature_only_results
                }

                print(f"\nClass {class_name} - Feature-Only Attack Results:")
                print(f"Success Rate: {feat_success_rate*100:.2f}%")
                print(f"Average Confidence: {feat_avg_confidence:.4f}")

        # Print overall results
        print("\n" + "="*50)
        print("Overall Attack Results")
        print("="*50)

        # Overall inversion attack results
        if attack_type in ['inversion', 'both'] and results['inversion_attack']:
            overall_inv_success = sum(res['success_rate'] for res in results['inversion_attack'].values()) / len(results['inversion_attack'])
            overall_inv_confidence = sum(res['avg_confidence'] for res in results['inversion_attack'].values()) / len(results['inversion_attack'])

            # Overall reconstruction quality
            overall_recon_quality = {}
            for metric in ['psnr', 'mse']:
                all_values = []
                for class_data in results['inversion_attack'].values():
                    if metric in class_data.get('avg_reconstruction_quality', {}):
                        all_values.append(class_data['avg_reconstruction_quality'][metric])
                if all_values:
                    overall_recon_quality[metric] = sum(all_values) / len(all_values)

            print(f"Inversion Attack - Overall Success Rate: {overall_inv_success*100:.2f}%")
            print(f"Inversion Attack - Overall Average Confidence: {overall_inv_confidence:.4f}")
            if 'psnr' in overall_recon_quality:
                print(f"Overall Average PSNR: {overall_recon_quality['psnr']:.2f} dB")
            if 'mse' in overall_recon_quality:
                print(f"Overall Average MSE: {overall_recon_quality['mse']:.4f}")

        # Overall feature-only attack results
        if attack_type in ['feature_only', 'both'] and results['feature_only_attack']:
            overall_feat_success = sum(res['success_rate'] for res in results['feature_only_attack'].values()) / len(results['feature_only_attack'])
            overall_feat_confidence = sum(res['avg_confidence'] for res in results['feature_only_attack'].values()) / len(results['feature_only_attack'])

            print(f"Feature-Only Attack - Overall Success Rate: {overall_feat_success*100:.2f}%")
            print(f"Feature-Only Attack - Overall Average Confidence: {overall_feat_confidence:.4f}")

        # Compare attack types if both were run
        if attack_type == 'both' and results['inversion_attack'] and results['feature_only_attack']:
            print("\nAttack Type Comparison:")
            if overall_inv_success > overall_feat_success:
                print(f"Inversion Attack outperforms Feature-Only Attack by {(overall_inv_success-overall_feat_success)*100:.2f}%")
            elif overall_feat_success > overall_inv_success:
                print(f"Feature-Only Attack outperforms Inversion Attack by {(overall_feat_success-overall_inv_success)*100:.2f}%")
            else:
                print(f"Both attack types perform equally with {overall_inv_success*100:.2f}% success rate")

        return results


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Feature Inversion Attack")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "imagenet", "mnist", "fashion_mnist"],
        help="Dataset to use (default: cifar10)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/autodl-tmp/datasets/classification",
        help="Directory for dataset storage (default: ./data)",
    )
    parser.add_argument(
        "--num_epochs", 
        type = int, 
        default = 5,
        help = "Number of epochs for training Q-Former (default: 5)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples per class to test (default: 3)",
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of PGD steps (default: 150)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Maximum perturbation range (default: 0.2)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="PGD step size (default: 0.01)"
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default="both",
        choices=["inversion", "feature_only", "both"],
        help="Type of attack to perform (default: both)",
    )
    parser.add_argument(
        "--save_images", action="store_true", help="Save reconstructed images"
    )
    args = parser.parse_args()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Initialize attack model
    print(f"Initializing Feature Inversion Attack for {args.dataset} dataset...")
    attack = FeatureInversionAttack(dataset_name=args.dataset, device=device)

    # Run attack experiment
    results = attack.run_attack_experiment(
        num_epochs=args.num_epochs,
        num_samples=args.num_samples,
        steps=args.steps,
        data_dir=args.data_dir,
        epsilon=args.epsilon,
        alpha=args.alpha,
        attack_type=args.attack_type,
        save_reconstructed_images=args.save_images,
    )

    # Save results
    import json

    output_file = f"results/attack_results_{args.dataset}_{args.attack_type}.json"

    # Convert results to serializable format
    serializable_results = {}

    for attack_type, attack_results in results.items():
        serializable_results[attack_type] = {}
        for class_name, class_data in attack_results.items():
            # Make a copy of class_data to avoid modifying the original
            serializable_class_data = {
                "success_rate": class_data["success_rate"],
                "avg_confidence": class_data["avg_confidence"],
            }

            # Add reconstruction quality metrics if available
            if "avg_reconstruction_quality" in class_data:
                serializable_class_data["avg_reconstruction_quality"] = class_data[
                    "avg_reconstruction_quality"
                ]

            # Process detailed results
            serializable_class_data["detailed_results"] = []
            for res in class_data["detailed_results"]:
                serializable_res = {
                    "success": res["success"],
                    "confidence": res["confidence"],
                    "probabilities": {k: v for k, v in res["probabilities"].items()},
                }

                # Add reconstruction quality if available
                if "reconstruction_quality" in res:
                    serializable_res["reconstruction_quality"] = res[
                        "reconstruction_quality"
                    ]

                serializable_class_data["detailed_results"].append(serializable_res)

            serializable_results[attack_type][class_name] = serializable_class_data

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=4)

    print(f"\nResults saved to '{output_file}'")


if __name__ == "__main__":
    main()
