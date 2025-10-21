import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import clip
from tqdm import tqdm
import os
from PIL import Image

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Custom TinyImageNet dataset
class TinyImageNet(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load data based on split
        if split == 'train':
            # Process train data with directory structure
            train_dir = os.path.join(root, 'train')
            self.classes = sorted(os.listdir(train_dir))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            
            for class_dir in self.classes:
                class_path = os.path.join(train_dir, class_dir, 'images')
                class_idx = self.class_to_idx[class_dir]
                
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.endswith('.JPEG'):
                            self.images.append(os.path.join(class_path, img_file))
                            self.labels.append(class_idx)
        
        elif split == 'val':
            # Process validation data
            val_dir = os.path.join(root, 'val')
            val_images_dir = os.path.join(val_dir, 'images')
            
            # Load val annotations file
            val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
            with open(val_annotations_file, 'r') as f:
                val_annotations = f.readlines()
            
            # Get class names and create mapping
            train_dir = os.path.join(root, 'train')
            self.classes = sorted(os.listdir(train_dir))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            
            # Parse annotations and add to the dataset
            for line in val_annotations:
                parts = line.strip().split('\t')
                img_file, class_id = parts[0], parts[1]
                if class_id in self.class_to_idx:
                    self.images.append(os.path.join(val_images_dir, img_file))
                    self.labels.append(self.class_to_idx[class_id])
                    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder in case of error
            return torch.zeros((3, 64, 64)), label


# 1. Function to extract intermediate features from CLIP model
def extract_clip_features(model, images):
    """Extract image features from CLIP model."""
    with torch.no_grad():
        features = model.encode_image(images)
    return features.to(torch.float)


# 2. Custom dataset for storing stolen features
class StolenFeatureDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.labels is not None:
            return feature, self.labels[idx]
        return feature


# 3. Q-Former inspired alignment module
class FeatureAlignmentModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes):
        super(FeatureAlignmentModule, self).__init__()
        # Multi-layer feature transformer
        self.feature_transformer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        transformed_features = self.feature_transformer(x)
        logits = self.classifier(transformed_features)
        return logits


# Define Q-Former architecture
class QFormer(nn.Module):
    def __init__(
        self,
        bert_model,
        img_feature_dim=512,
        text_feature_dim=768,
        num_classes=200,
    ):
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
            similarity = F.cosine_similarity(
                projected_img_features, text_features, dim=1
            )
            return similarity, self.classifier(projected_img_features)
        else:
            # Classification only
            return self.classifier(projected_img_features)


# 4. Main attack implementation
class InversionAttack:
    def __init__(self, feature_dim=512, hidden_dim=768, num_classes=200, class_names=None, device="cuda"):
        self.alignment_model = self._init_q_former(feature_dim, hidden_dim, num_classes).to(device).to(torch.float)
        self.class_names = class_names
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizer.from_pretrained(
            "/root/autodl-tmp/models/google-bert/bert-base-uncased"
        )
        self.optimizer = optim.AdamW(self.alignment_model.parameters(), lr=1e-4)
        self.device = device

    def _init_q_former(self, feature_dim=512, hidden_dim=768, num_classes=200):
        """Initialize Q-Former module for feature-text alignment"""
        # Use pre-trained BERT as the base for Q-Former
        bert_model = BertModel.from_pretrained(
            "/root/autodl-tmp/models/google-bert/bert-base-uncased"
        )
        return QFormer(bert_model, img_feature_dim=feature_dim, text_feature_dim=hidden_dim, num_classes=num_classes)

    def train(self, train_loader, val_loader=None, epochs=30):
        best_acc = 0.0
        train_losses = []
        val_accs = []

        # Generate simple text descriptions for all classes
        text_descriptions = [f"a photo of a {cls}" for cls in self.class_names]
        text_inputs = self.tokenizer(
            text_descriptions, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        for epoch in range(epochs):
            # Training phase
            self.alignment_model.train()
            running_loss = 0.0

            for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                features, labels = features.to(self.device), labels.to(self.device)

                # Create text inputs for corresponding labels
                batch_text_inputs = {
                    k: torch.cat(
                        [text_inputs[k][labels[i]].unsqueeze(0) for i in range(len(labels))],
                        dim=0,
                    )
                    for k in text_inputs.keys()
                }

                # Forward pass
                self.optimizer.zero_grad()

                similarity, logits = self.alignment_model(features, text=batch_text_inputs)
                classification_loss = self.criterion(logits, labels)
                contrastive_loss = -torch.mean(similarity)
                loss = classification_loss + 0.5 * contrastive_loss

                # Backward pass
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}")

            # Validation phase
            if val_loader:
                val_acc = self.evaluate(val_loader)
                val_accs.append(val_acc)
                print(f"Validation Accuracy: {val_acc:.4f}")

                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(
                        self.alignment_model.state_dict(),
                        "/root/autodl-tmp/models/FeatureInversionClassification/best_tinyimagenet_inversion_model_vit32.pth",
                    )
                    print(f"Model saved with accuracy: {best_acc:.4f}")

        # Plot training curves
        self.plot_training_curves(train_losses, val_accs)
        return train_losses, val_accs

    def evaluate(self, data_loader, return_top5=False):
        self.alignment_model.eval()
        correct = 0
        correct_top5 = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            text_descriptions = [f"a photo of a {cls}" for cls in self.class_names]
            text_inputs = self.tokenizer(
                text_descriptions, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            for features, labels in data_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                # Create text inputs for corresponding labels
                batch_text_inputs = {
                    k: torch.cat(
                        [text_inputs[k][labels[i]].unsqueeze(0) for i in range(len(labels))],
                        dim=0,
                    )
                    for k in text_inputs.keys()
                }

                _, outputs = self.alignment_model(features, batch_text_inputs)
                _, predicted = torch.max(outputs.data, 1)

                #         total += labels.size(0)
                #         correct += (predicted == labels).sum().item()

                #         all_preds.extend(predicted.cpu().numpy())
                #         all_labels.extend(labels.cpu().numpy())

                # accuracy = correct / total
                # return accuracy
                # Top-5 accuracy calculation
                _, top5_indices = torch.topk(outputs.data, 5, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Calculate Top-5 accuracy
                for i, label in enumerate(labels):
                    if label in top5_indices[i]:
                        correct_top5 += 1

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        top5_accuracy = correct_top5 / total

        if return_top5:
            return accuracy, top5_accuracy
        return accuracy

    def test(self, test_loader):
        # Load best model
        try:
            self.alignment_model.load_state_dict(
                torch.load(
                    "/root/autodl-tmp/models/FeatureInversionClassification/best_tinyimagenet_inversion_model_vit32.pth"
                )
            )
            print("Loaded best model for testing")
        except:
            print("Using current model for testing")

        # self.alignment_model.eval()
        # all_preds = []
        # all_labels = []
        self.alignment_model.eval()
        all_preds = []
        all_labels = []
        correct_top5 = 0
        total = 0
        top5_class_correct = np.zeros(self.num_classes)
        top5_class_total = np.zeros(self.num_classes)

        with torch.no_grad():
            text_descriptions = [f"a photo of a {cls}" for cls in self.class_names]
            text_inputs = self.tokenizer(
                text_descriptions, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            for features, labels in tqdm(test_loader, desc="Testing"):
                features, labels = features.to(self.device), labels.to(self.device)

                # Create text inputs for corresponding labels
                batch_text_inputs = {
                    k: torch.cat(
                        [text_inputs[k][labels[i]].unsqueeze(0) for i in range(len(labels))],
                        dim=0,
                    )
                    for k in text_inputs.keys()
                }

                _, outputs = self.alignment_model(features, batch_text_inputs)
                _, predicted = torch.max(outputs.data, 1)

                #         all_preds.extend(predicted.cpu().numpy())
                #         all_labels.extend(labels.cpu().numpy())

                # # Calculate metrics
                # accuracy = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
                # print(f"Test Accuracy: {accuracy:.4f}")

                # # Generate classification report
                # report = classification_report(all_labels, all_preds)
                # print("\nClassification Report:")
                # print(report)

                # # Plot confusion matrix (only for top classes due to size)
                # self.plot_confusion_matrix(all_labels, all_preds, max_classes=20)

                # return accuracy, all_preds, all_labels

                # Top-5 predictions
                _, top5_indices = torch.topk(outputs.data, 5, dim=1)

                total += labels.size(0)

                # Calculate Top-5 accuracy overall and per class
                for i, label in enumerate(labels):
                    label_idx = label.item()
                    top5_class_total[label_idx] += 1

                    if label in top5_indices[i]:
                        correct_top5 += 1
                        top5_class_correct[label_idx] += 1

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        top1_accuracy = sum(np.array(all_preds) == np.array(all_labels)) / len(
            all_labels
        )
        top5_accuracy = correct_top5 / total

        print(f"Test Accuracy - Top-1: {top1_accuracy:.4f}, Top-5: {top5_accuracy:.4f}")

        # Generate classification report for Top-1
        report = classification_report(all_labels, all_preds)
        print("\nClassification Report (Top-1):")
        print(report)

        # Plot confusion matrix
        self.plot_confusion_matrix(all_labels, all_preds)

        # Per-class accuracy analysis for Top-1
        top1_class_correct = np.zeros(self.num_classes)
        top1_class_total = np.zeros(self.num_classes)

        for i in range(len(all_labels)):
            class_idx = all_labels[i]
            top1_class_total[class_idx] += 1
            if all_preds[i] == all_labels[i]:
                top1_class_correct[class_idx] += 1

        print("\nPer-class Top-1 Accuracy:")
        for i in range(self.num_classes):
            print(
                f"{self.class_names[i]}: {100 * top1_class_correct[i] / top1_class_total[i]:.2f}%"
            )

        print("\nPer-class Top-5 Accuracy:")
        for i in range(self.num_classes):
            print(
                f"{self.class_names[i]}: {100 * top5_class_correct[i] / top5_class_total[i]:.2f}%"
            )

        # Plot Top-1 vs Top-5 accuracy comparison by class
        self.plot_top1_vs_top5_accuracy(
            top1_class_correct, top1_class_total, top5_class_correct, top5_class_total
        )

        return top1_accuracy, top5_accuracy, all_preds, all_labels

    def plot_training_curves(self, train_losses, val_accs):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        if val_accs:
            plt.subplot(1, 2, 2)
            plt.plot(val_accs)
            plt.title("Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")

        plt.tight_layout()
        plt.savefig("training_curves_tinyimagenet_vit32.png")
        plt.show()

    def plot_confusion_matrix(self, true_labels, pred_labels, max_classes=20):
        # Get most common classes for visualization (full 200x200 matrix would be too large)
        unique_labels, counts = np.unique(true_labels, return_counts=True)
        top_indices = np.argsort(-counts)[:max_classes]  # Get indices of top classes

        # Filter to only include top classes
        mask = np.isin(true_labels, top_indices) & np.isin(pred_labels, top_indices)
        filtered_true = np.array(true_labels)[mask]
        filtered_pred = np.array(pred_labels)[mask]

        # Create confusion matrix for top classes
        cm = confusion_matrix(filtered_true, filtered_pred)
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix (Top {max_classes} Classes)")
        plt.colorbar()

        # Get class names for the top classes
        top_class_names = [self.class_names[idx] for idx in top_indices]

        tick_marks = np.arange(len(top_class_names))
        plt.xticks(tick_marks, top_class_names, rotation=90)
        plt.yticks(tick_marks, top_class_names)

        fmt = "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14,
                )

        plt.ylabel("True Label", fontsize = 24)
        plt.xlabel("Predicted Label", fontsize=24)
        plt.tight_layout()
        plt.savefig("confusion_matrix_tinyimagenet_vit32.png")
        plt.show()

    def plot_top1_vs_top5_accuracy(
        self, top1_correct, top1_total, top5_correct, top5_total
    ):
        """Plot comparison between Top-1 and Top-5 accuracy for each class"""
        top1_accuracy = 100 * top1_correct / top1_total
        top5_accuracy = 100 * top5_correct / top5_total

        fig, ax = plt.figure(figsize=(12, 6)), plt.axes()
        x = np.arange(len(self.class_names))
        width = 0.35

        # Plot bars
        bars1 = ax.bar(x - width / 2, top1_accuracy, width, label="Top-1 Accuracy")
        bars2 = ax.bar(x + width / 2, top5_accuracy, width, label="Top-5 Accuracy")

        # Add labels and title
        ax.set_xlabel("Classes")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Top-1 vs Top-5 Accuracy by Class")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [name.capitalize() for name in self.class_names], rotation=45
        )
        ax.legend()

        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        autolabel(bars1)
        autolabel(bars2)

        plt.tight_layout()
        plt.savefig("top1_vs_top5_accuracy_tinyimagenet_vit32.png")
        plt.show()


# Main execution flow
def run_inversion_attack():
    # 1. Define paths and load TinyImageNet dataset
    data_root = "/root/autodl-tmp/datasets/image_caption_generation/tiny-imagenet-200"  # Update with your path

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(224),  # CLIP requires 224x224 images
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load datasets
    print("Loading TinyImageNet datasets...")
    train_dataset = TinyImageNet(root=data_root, split='train', transform=transform)
    val_dataset = TinyImageNet(root=data_root, split='val', transform=transform)

    # Get class names
    class_names = train_dataset.classes
    print(f"TinyImageNet has {len(class_names)} classes")
    print(f"Sample class names: {class_names[:5]}...")

    # 2. Load CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)

    # 3. Extract features from TinyImageNet (simulating stolen features)
    print("Extracting CLIP features from training set...")
    train_features = []
    train_labels = []
    batch_size = 64
    train_loader_original = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    for images, labels in tqdm(train_loader_original):
        images = images.to(device)
        features = extract_clip_features(clip_model, images)
        train_features.append(features.cpu())
        train_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    print("Extracting CLIP features from validation set...")
    val_features = []
    val_labels = []
    val_loader_original = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    for images, labels in tqdm(val_loader_original):
        images = images.to(device)
        features = extract_clip_features(clip_model, images)
        val_features.append(features.cpu())
        val_labels.append(labels)

    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    # 4. Create datasets for stolen features
    train_dataset = StolenFeatureDataset(train_features, train_labels)
    val_dataset = StolenFeatureDataset(val_features, val_labels)

    # Split training set into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # 5. Create data loaders for the stolen features
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"Feature dimension: {train_features.shape[1]}")

    # 6. Initialize and run the attack
    attack = InversionAttack(
        feature_dim=train_features.shape[1], 
        hidden_dim=768, 
        num_classes=len(class_names), 
        class_names=class_names,
        device=device
    )

    print("Starting attack training...")
    attack.train(train_loader, val_loader, epochs=50)

    # 7. Test the attack on the validation set
    print("\nEvaluating attack on validation set...")
    # acc, preds, labels = attack.test(test_loader)
    top1_acc, top5_acc, preds, labels = attack.test(test_loader)

    print(f"\nOverall Results:")
    print(f"Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"Improvement from Top-1 to Top-5: {(top5_acc-top1_acc)*100:.2f}%")

    # 8. Per-class accuracy analysis (for top classes)
    print("\nPer-class accuracy (top 20 classes):")
    unique_labels, counts = np.unique(labels, return_counts=True)
    top_indices = np.argsort(-counts)[:20]  # Get indices of top 20 classes

    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))

    for i in range(len(labels)):
        class_idx = labels[i]
        class_total[class_idx] += 1
        if preds[i] == labels[i]:
            class_correct[class_idx] += 1

    for idx in top_indices:
        if class_total[idx] > 0:
            print(f"{class_names[idx]}: {100 * class_correct[idx] / class_total[idx]:.2f}%")

    return attack


if __name__ == "__main__":
    run_inversion_attack()
