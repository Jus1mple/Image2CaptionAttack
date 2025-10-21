import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import clip
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


# 4. Main attack implementation
class InversionAttack:
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=10, device = "cuda"):
        self.alignment_model = FeatureAlignmentModule(
            feature_dim, hidden_dim, num_classes
        ).to(device).to(torch.float)

        self.class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.alignment_model.parameters(), lr=1e-4)
        self.device = device

    def train(self, train_loader, val_loader=None, epochs=30):
        best_acc = 0.0
        train_losses = []
        val_accs = []
        text_descriptions = [f"a photo of a {cls}" for cls in self.class_names]

        for epoch in range(epochs):
            # Training phase
            self.alignment_model.train()
            running_loss = 0.0

            for features, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs}"
            ):
                features, labels = features.to(device), labels.to(device)

                # Forward pass
                self.optimizer.zero_grad()

                # outputs = self.alignment_model(features)
                logits = self.alignment_model(
                    features
                )
                loss = self.criterion(logits, labels)
                

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
                        self.alignment_model.state_dict(), "/root/autodl-tmp/models/FeatureInversionClassification/best_inversion_model.pth"
                    )
                    print(f"Model saved with accuracy: {best_acc:.4f}")

        # Plot training curves
        self.plot_training_curves(train_losses, val_accs)
        return train_losses, val_accs

    def evaluate(self, data_loader):
        self.alignment_model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in data_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = self.alignment_model(features)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        return accuracy

    def test(self, test_loader):
        # Load best model
        try:
            self.alignment_model.load_state_dict(
                torch.load(
                    "/root/autodl-tmp/models/FeatureInversionClassification/best_inversion_model.pth"
                )
            )
            print("Loaded best model for testing")
        except:
            print("Using current model for testing")

        self.alignment_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="Testing"):
                features, labels = features.to(device), labels.to(device)
                outputs = self.alignment_model(features)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Generate classification report
        report = classification_report(all_labels, all_preds)
        print("\nClassification Report:")
        print(report)

        # Plot confusion matrix
        self.plot_confusion_matrix(all_labels, all_preds)

        return accuracy, all_preds, all_labels

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
        plt.savefig("training_curves.png")
        plt.show()

    def plot_confusion_matrix(self, true_labels, pred_labels):
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        # classes = [f"Class {i}" for i in range(10)]
        classes = [name.capitalize() for name in class_names]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

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
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.show()


# Main execution flow
def run_inversion_attack():
    # 1. Load CIFAR-10 dataset
    transform = transforms.Compose(
        [
            transforms.Resize(224),  # CLIP requires 224x224 images
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    cifar_train = torchvision.datasets.CIFAR10(
        root="/root/autodl-tmp/datasets/classification/cifar10",
        train=True,
        download=True,
        transform=transform,
    )
    cifar_test = torchvision.datasets.CIFAR10(
        root="/root/autodl-tmp/datasets/classification/cifar10",
        train=False,
        download=True,
        transform=transform,
    )

    # Class names for interpretation
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    print(f"CIFAR-10 classes: {class_names}")

    # 2. Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 3. Extract features from CIFAR-10 (simulating stolen features)
    print("Extracting CLIP features from training set...")
    train_features = []
    train_labels = []
    batch_size = 64
    train_loader_original = DataLoader(
        cifar_train, batch_size=batch_size, shuffle=False
    )

    for images, labels in tqdm(train_loader_original):
        images = images.to(device)
        features = extract_clip_features(clip_model, images)
        train_features.append(features.cpu())
        train_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    print("Extracting CLIP features from test set...")
    test_features = []
    test_labels = []
    test_loader_original = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

    for images, labels in tqdm(test_loader_original):
        images = images.to(device)
        features = extract_clip_features(clip_model, images)
        test_features.append(features.cpu())
        test_labels.append(labels)

    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # 4. Create datasets for stolen features
    train_dataset = StolenFeatureDataset(train_features, train_labels)
    test_dataset = StolenFeatureDataset(test_features, test_labels)

    # Split training set into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # 5. Create data loaders for the stolen features
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Feature dimension: {train_features.shape[1]}")

    # 6. Initialize and run the attack
    attack = InversionAttack(
        feature_dim=train_features.shape[1], hidden_dim=256, num_classes=10, device = device
    )
    print("Starting attack training...")
    attack.train(train_loader, val_loader, epochs=50)

    # 7. Test the attack on the full test set
    print("\nEvaluating attack on full test set...")
    acc, preds, labels = attack.test(test_loader)

    # 8. Per-class accuracy analysis
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    for i in range(len(labels)):
        class_idx = labels[i]
        class_total[class_idx] += 1
        if preds[i] == labels[i]:
            class_correct[class_idx] += 1

    print("\nPer-class accuracy:")
    for i in range(10):
        print(f"{class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")

    return attack


if __name__ == "__main__":
    run_inversion_attack()
