import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel, BertModel, BertTokenizer
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader
import os
import random
import json
import time


class FeatureInversionAttack:
    """
    实现基于窃取的中间层特征的反转攻击，使用PGD算法和特征-文本对齐
    专门为CIFAR10数据集设计，测试整个测试集
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # 初始化模型组件
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 冻结原始CLIP模型参数
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # CIFAR10类别
        self.cifar10_classes = [
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

        # 初始化Q-Former特征-文本对齐模块
        self.q_former = self._init_q_former().to(device)

        # 用于转换CIFAR10图像的变换
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),  # CLIP需要224x224的输入
                transforms.ToTensor(),
            ]
        )

        # 创建结果目录
        os.makedirs("results", exist_ok=True)

    def _init_q_former(self):
        """初始化Q-Former模块，用于特征-文本对齐，为CIFAR10定制"""
        # 使用预训练的BERT作为Q-Former的基础
        bert_model = BertModel.from_pretrained("bert-base-uncased")

        # 定义Q-Former架构
        class QFormer(nn.Module):
            def __init__(
                self,
                bert_model,
                img_feature_dim=512,
                text_feature_dim=768,
                num_classes=10,
            ):
                super(QFormer, self).__init__()
                self.bert = bert_model
                self.img_projection = nn.Linear(img_feature_dim, text_feature_dim)
                self.classifier = nn.Linear(
                    text_feature_dim, num_classes
                )  # CIFAR10有10个类别

            def forward(self, img_features, text=None):
                # 投影图像特征到文本空间
                projected_img_features = self.img_projection(img_features)

                if text is not None:
                    # 文本特征提取
                    outputs = self.bert(**text)
                    text_features = outputs.last_hidden_state[:, 0, :]  # 使用[CLS]标记

                    # 计算图像和文本特征之间的相似度
                    similarity = F.cosine_similarity(
                        projected_img_features, text_features, dim=1
                    )
                    return similarity, self.classifier(projected_img_features)
                else:
                    # 仅进行分类
                    return self.classifier(projected_img_features)

        return QFormer(bert_model, img_feature_dim=512, num_classes=10)

    def load_cifar10(self, batch_size=64, train=True, full_test=True):
        """加载CIFAR10数据集并预处理"""
        print(f"Loading CIFAR10 {'training' if train else 'test'} dataset...")

        dataset = torchvision.datasets.CIFAR10(
            root="/root/autodl-tmp/datasets/classification/cifar10", train=train, download=True, transform=self.transform
        )

        if not full_test and not train:
            # 如果不需要完整测试集且在测试模式下，则使用一个较小的子集
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            subset_indices = indices[:1000]  # 使用1000个样本
            from torch.utils.data import Subset

            dataset = Subset(dataset, subset_indices)

        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,  # 训练时打乱数据，测试时不打乱
            num_workers=4,
            pin_memory=True,
        )

        return dataloader

    def steal_features(self, image):
        """从CLIP模型中窃取中间层特征"""
        if isinstance(image, np.ndarray):
            # 将numpy数组转换为PIL图像
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # 将PyTorch张量转换为PIL图像
            if image.dim() == 4:  # 批处理
                image = image[0]  # 取第一个图像
            image = transforms.ToPILImage()(image)

        # 确保图像是RGB格式
        image = image.convert("RGB")

        # 处理图像
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 获取CLIP模型中的图像特征
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        return image_features

    def steal_features_batch(self, images, labels=None):
        """批量窃取多张图像的特征"""
        batch_size = images.size(0)

        # 调整图像大小并转换为合适的格式
        processed_images = []
        for i in range(batch_size):
            img = transforms.ToPILImage()(images[i])
            img = img.convert("RGB")
            processed_images.append(img)

        # 使用CLIP处理器批量处理图像
        inputs = self.processor(images=processed_images, return_tensors="pt").to(
            self.device
        )

        # 获取CLIP特征
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        if labels is not None:
            labels = labels.to(self.device)
            return image_features, labels
        else:
            return image_features

    def save_model(self, path="models/q_former.pth"):
        """保存Q-Former模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_former.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="models/q_former.pth"):
        """加载Q-Former模型"""
        if os.path.exists(path):
            self.q_former.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")
            return True
        else:
            print(f"Model file {path} not found")
            return False

    def train_q_former(
        self, train_loader, val_loader=None, epochs=10, save_path="models/q_former.pth"
    ):
        """
        使用CIFAR10数据集训练Q-Former模块

        参数:
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器(可选)
        - epochs: 训练轮数
        - save_path: 模型保存路径
        """
        print("Preparing CIFAR10 text descriptions...")
        # 准备CIFAR10类别的文本描述
        text_descriptions = [f"a photo of a {cls}" for cls in self.cifar10_classes]

        # 处理文本描述
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_inputs = tokenizer(
            text_descriptions, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # 设置优化器
        optimizer = optim.Adam(self.q_former.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=2, factor=0.5
        )

        # 记录训练历史
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        best_val_acc = 0.0

        print(f"Starting Q-Former training for {epochs} epochs...")
        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.q_former.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0

            for images, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"
            ):
                # 将图像和标签移到设备上
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 窃取特征
                stolen_features, _ = self.steal_features_batch(images, labels)

                # 重置优化器梯度
                optimizer.zero_grad()

                # 创建对应标签的文本输入
                batch_text_inputs = {
                    k: torch.cat(
                        [
                            text_inputs[k][labels[i]].unsqueeze(0)
                            for i in range(len(labels))
                        ],
                        dim=0,
                    )
                    for k in text_inputs.keys()
                }

                # 前向传播
                similarity, logits = self.q_former(stolen_features, batch_text_inputs)

                # 计算损失: 分类损失 + 对比损失
                classification_loss = criterion(logits, labels)
                contrastive_loss = -torch.mean(similarity)  # 最大化相似性

                loss = classification_loss + 0.5 * contrastive_loss

                # 反向传播
                loss.backward()
                optimizer.step()

                # 计算精度
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                epoch_loss += loss.item()
                batch_count += 1

            train_loss = epoch_loss / batch_count
            train_acc = 100.0 * correct / total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
            )

            # 验证阶段
            if val_loader:
                val_loss, val_acc = self.evaluate_q_former(
                    val_loader, criterion, text_inputs
                )
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # 更新学习率调度器
                scheduler.step(val_loss)

                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(save_path)
                    print(
                        f"New best model saved with validation accuracy: {val_acc:.2f}%"
                    )
            else:
                # 如果没有验证集，则每个epoch保存模型
                self.save_model(save_path)

        # 如果有验证集但未保存任何模型，则保存最后一个模型
        if val_loader and best_val_acc == 0.0:
            self.save_model(save_path)

        # 绘制训练历史
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        if val_loader:
            plt.plot(history["val_loss"], label="Val Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="Train Acc")
        if val_loader:
            plt.plot(history["val_acc"], label="Val Acc")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()
        plt.savefig("results/training_history.png")
        print("Training history plot saved to results/training_history.png")

        return history

    def evaluate_q_former(self, dataloader, criterion=None, text_inputs=None):
        """评估Q-Former在CIFAR10上的性能"""
        self.q_former.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        batch_count = 0

        # 如果没有提供criterion和text_inputs，则初始化它们
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        if text_inputs is None:
            # 准备CIFAR10类别的文本描述
            text_descriptions = [f"a photo of a {cls}" for cls in self.cifar10_classes]
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            text_inputs = tokenizer(
                text_descriptions, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

        class_correct = [0] * 10
        class_total = [0] * 10

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                # 将图像和标签移到设备上
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 窃取特征
                stolen_features, _ = self.steal_features_batch(images, labels)

                # 创建对应标签的文本输入
                batch_text_inputs = {
                    k: torch.cat(
                        [
                            text_inputs[k][labels[i]].unsqueeze(0)
                            for i in range(len(labels))
                        ],
                        dim=0,
                    )
                    for k in text_inputs.keys()
                }

                # 前向传播
                similarity, logits = self.q_former(stolen_features, batch_text_inputs)

                # 计算损失
                classification_loss = criterion(logits, labels)
                contrastive_loss = -torch.mean(similarity)
                loss = classification_loss + 0.5 * contrastive_loss

                total_loss += loss.item()
                batch_count += 1

                # 计算整体精度
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 计算每个类别的精度
                for i in range(len(labels)):
                    label = labels[i]
                    pred = predicted[i]
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

        # 计算平均损失和整体精度
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0

        print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 打印每个类别的精度
        print("\nPer-Class Accuracy:")
        for i in range(10):
            class_acc = (
                100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            )
            print(f"{self.cifar10_classes[i]}: {class_acc:.2f}%")

        return avg_loss, accuracy

    def pgd_attack(
        self, target_features, target_class, epsilon=0.1, alpha=0.01, steps=100
    ):
        """
        使用PGD算法反演窃取的特征，生成可能的图像表示

        参数:
        - target_features: 目标(窃取的)特征
        - target_class: 目标类别
        - epsilon: PGD扰动的最大范围
        - alpha: PGD步长
        - steps: PGD迭代步数

        返回:
        - 反演的图像特征
        """
        # 初始化一个随机图像特征作为起点
        x = torch.randn_like(target_features).requires_grad_(True)

        # 准备目标类别的文本描述
        text_description = f"a photo of a {self.cifar10_classes[target_class]}"

        # 处理文本描述
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_input = tokenizer(
            [text_description], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # 设置优化器
        optimizer = optim.Adam([x], lr=alpha)

        # 原始特征范围
        x_orig = x.clone().detach()

        # 保存损失历史以便绘图
        loss_history = []

        # PGD攻击循环
        for i in range(steps):
            # 重置梯度
            optimizer.zero_grad()

            # 计算特征距离损失
            feature_loss = F.mse_loss(x, target_features)

            # 计算分类损失
            logits = self.q_former(x)
            target = torch.tensor([target_class]).to(self.device)
            classification_loss = F.cross_entropy(logits, target)

            # 计算总损失
            loss = feature_loss + classification_loss
            loss_history.append(loss.item())

            # 反向传播
            loss.backward()

            # 更新步骤
            optimizer.step()

            # 投影回ε-球
            with torch.no_grad():
                delta = x - x_orig
                delta = torch.clamp(delta, -epsilon, epsilon)
                x.data = x_orig + delta

        return x.detach(), loss_history

    def evaluate_attack(self, inverted_features, ground_truth_class):
        """评估攻击效果"""
        self.q_former.eval()
        with torch.no_grad():
            logits = self.q_former(inverted_features)
            probabilities = F.softmax(logits, dim=1)

            # 获取所有类别的概率
            probs_dict = {
                self.cifar10_classes[i]: probabilities[0, i].item() for i in range(10)
            }

            # 获取预测类别
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        success = predicted_class == ground_truth_class

        return success, confidence, probs_dict, predicted_class

    def test_on_full_dataset(
        self, dataloader, num_samples_per_class=None, steps=100, epsilon=0.1, alpha=0.01
    ):
        """
        在完整测试集上测试攻击效果

        参数:
        - dataloader: 测试数据加载器
        - num_samples_per_class: 每个类别测试的样本数，如果为None则测试所有样本
        - steps: PGD攻击的步数
        - epsilon: PGD扰动的最大范围
        - alpha: PGD步长

        返回:
        - 攻击结果字典
        """
        print("Starting full test set evaluation...")

        # 按类别收集样本
        class_samples = {i: [] for i in range(10)}
        all_images = []
        all_labels = []

        print("Collecting samples by class...")
        for images, labels in tqdm(dataloader):
            for i, (img, label) in enumerate(zip(images, labels)):
                label_idx = label.item()
                # 如果指定了每个类别的样本数，并且已经收集了足够的样本，则跳过
                if (
                    num_samples_per_class is not None
                    and len(class_samples[label_idx]) >= num_samples_per_class
                ):
                    continue

                class_samples[label_idx].append((img, i))
                all_images.append(img)
                all_labels.append(label_idx)

        # 检查是否所有类别都有足够的样本
        if num_samples_per_class is not None:
            for cls_idx, samples in class_samples.items():
                if len(samples) < num_samples_per_class:
                    print(
                        f"Warning: Class {self.cifar10_classes[cls_idx]} only has {len(samples)} samples, less than requested {num_samples_per_class}"
                    )

        # 整体结果
        results = {
            "overall": {
                "total_samples": 0,
                "successful_attacks": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_steps": 0,
            },
            "per_class": {
                cls: {
                    "total_samples": 0,
                    "successful_attacks": 0,
                    "success_rate": 0.0,
                    "avg_confidence": 0.0,
                    "confusion_matrix": {
                        other_cls: 0 for other_cls in self.cifar10_classes
                    },
                }
                for cls in self.cifar10_classes
            },
        }

        # 混淆矩阵
        confusion_matrix = np.zeros((10, 10), dtype=int)

        # 对每个类别进行攻击
        total_samples = 0
        successful_attacks = 0
        total_confidence = 0.0

        print("Running attacks...")
        for cls_idx in range(10):
            cls_name = self.cifar10_classes[cls_idx]
            samples = class_samples[cls_idx]

            # 如果指定了每个类别的样本数，则只取那么多样本
            if num_samples_per_class is not None:
                samples = samples[:num_samples_per_class]

            print(f"\nTesting class {cls_name} with {len(samples)} samples")

            # 记录该类别的结果
            class_successful = 0
            class_confidence_sum = 0.0

            for img, _ in tqdm(samples, desc=f"Class {cls_name}"):
                # 窃取特征
                target_features = self.steal_features(img)

                # 执行PGD攻击
                inverted_features, _ = self.pgd_attack(
                    target_features, cls_idx, epsilon=epsilon, alpha=alpha, steps=steps
                )

                # 评估攻击效果
                success, confidence, _, predicted_class = self.evaluate_attack(
                    inverted_features, cls_idx
                )

                # 更新结果
                total_samples += 1
                results["overall"]["total_samples"] += 1
                results["per_class"][cls_name]["total_samples"] += 1

                if success:
                    successful_attacks += 1
                    class_successful += 1
                    results["overall"]["successful_attacks"] += 1
                    results["per_class"][cls_name]["successful_attacks"] += 1

                # 更新混淆矩阵
                confusion_matrix[cls_idx][predicted_class] += 1
                results["per_class"][cls_name]["confusion_matrix"][
                    self.cifar10_classes[predicted_class]
                ] += 1

                # 累计置信度
                total_confidence += confidence
                class_confidence_sum += confidence

            # 计算该类别的平均结果
            class_samples_count = len(samples)
            class_success_rate = (
                class_successful / class_samples_count if class_samples_count > 0 else 0
            )
            class_avg_confidence = (
                class_confidence_sum / class_samples_count
                if class_samples_count > 0
                else 0
            )

            results["per_class"][cls_name]["success_rate"] = class_success_rate
            results["per_class"][cls_name]["avg_confidence"] = class_avg_confidence

            print(
                f"Class {cls_name} - Success Rate: {class_success_rate:.2%}, Avg Confidence: {class_avg_confidence:.4f}"
            )

        # 计算整体结果
        overall_success_rate = (
            successful_attacks / total_samples if total_samples > 0 else 0
        )
        overall_avg_confidence = (
            total_confidence / total_samples if total_samples > 0 else 0
        )

        results["overall"]["success_rate"] = overall_success_rate
        results["overall"]["avg_confidence"] = overall_avg_confidence
        results["overall"]["avg_steps"] = steps

        print("\nOverall Results:")
        print(f"Total Samples: {total_samples}")
        print(f"Successful Attacks: {successful_attacks}")
        print(f"Success Rate: {overall_success_rate:.2%}")
        print(f"Average Confidence: {overall_avg_confidence:.4f}")

        # 保存混淆矩阵
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, self.cifar10_classes, rotation=45)
        plt.yticks(tick_marks, self.cifar10_classes)

        # 在每个单元格中添加文本
        thresh = confusion_matrix.max() / 2.0
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(
                    j,
                    i,
                    format(confusion_matrix[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black",
                )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig("results/confusion_matrix.png")

        # 保存结果为JSON
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f"results/attack_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=4)

        return results, confusion_matrix



def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 初始化攻击模型
    attack = FeatureInversionAttack(device=device)

    # 加载数据集
    train_loader = attack.load_cifar10(batch_size=64, train=True)
    test_loader = attack.load_cifar10(batch_size=64, train=False, full_test=True)

    # 检查是否有预训练的模型
    model_path = "models/q_former.pth"
    if not attack.load_model(model_path):
        print("No pre-trained model found. Training a new one...")
        # 训练Q-Former
        attack.train_q_former(train_loader, test_loader, epochs=5, save_path=model_path)

    # 测试模型在测试集上的性能
    print("\nEvaluating Q-Former on test set...")
    attack.evaluate_q_former(test_loader)

    # 在完整测试集上测试攻击
    # 为了速度，可以每个类别只测试部分样本
    samples_per_class = 100  # 每个类别测试的样本数，设置为None则测试所有样本
    results, confusion_matrix = attack.test_on_full_dataset(
        test_loader,
        num_samples_per_class=samples_per_class,
        steps=100,  # PGD迭代步数
        epsilon=0.2,  # 扰动范围
        alpha=0.01,  # 步长
    )

    print("\nAttack testing completed.")
    print(f"Results saved to results/attack_results_*.json")
    print(f"Confusion matrix saved to results/confusion_matrix.png")


if __name__ == "__main__":
    main()
