"""
Data utilities for Garbage Classification System
包含数据加载、增强、加权采样等核心功能
"""

import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from PIL import Image
from collections import Counter

# 尝试导入高级增强模块
try:
    from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment
    HAS_AUTOAUGMENT = True
except ImportError:
    HAS_AUTOAUGMENT = False
    print("Warning: AutoAugment/RandAugment not available, falling back to basic augmentation.")

# 从 config 导入配置
# 使用 getattr 或 try-except 模式来防止因 config 缺失某些参数导致的 Import Error
import config

DATA_DIR = getattr(config, 'DATA_DIR', './data')
TRAIN_DIR = getattr(config, 'TRAIN_DIR', './data/train')
VAL_DIR = getattr(config, 'VAL_DIR', './data/val')
IMAGE_SIZE = getattr(config, 'IMAGE_SIZE', 224)
BATCH_SIZE = getattr(config, 'BATCH_SIZE', 32)
NUM_CLASSES = getattr(config, 'NUM_CLASSES', 44)
CLASS_NAMES = getattr(config, 'CLASS_NAMES', [])

# 高级训练配置 (带默认值)
USE_AUTOAUGMENT = getattr(config, 'USE_AUTOAUGMENT', False)
USE_RANDAUGMENT = getattr(config, 'USE_RANDAUGMENT', False)
RANDAUGMENT_N = getattr(config, 'RANDAUGMENT_N', 2)
RANDAUGMENT_M = getattr(config, 'RANDAUGMENT_M', 9)
USE_EFFECTIVE_NUMBER_WEIGHTING = getattr(config, 'USE_EFFECTIVE_NUMBER_WEIGHTING', True)
EFFECTIVE_NUMBER_BETA = getattr(config, 'EFFECTIVE_NUMBER_BETA', 0.9999)
USE_WEIGHTED_SAMPLER = getattr(config, 'USE_WEIGHTED_SAMPLER', False)
HIGH_CONFUSION_CLASSES = getattr(config, 'HIGH_CONFUSION_CLASSES', [])
LONG_TAIL_CLASSES = getattr(config, 'LONG_TAIL_CLASSES', [])
CONFUSION_CLASS_BOOST = getattr(config, 'CONFUSION_CLASS_BOOST', 1.2)
LONG_TAIL_BOOST = getattr(config, 'LONG_TAIL_BOOST', 1.5)


# ============================================================================
# 1. 核心工具函数
# ============================================================================

def get_class_mapping(data_dir):
    """
    Get class mapping from dataset directory.
    这是 app.py 依赖的核心函数。
    """
    try:
        if os.path.exists(data_dir):
            # 获取子目录作为类名
            classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            classes.sort()
            if not classes:
                # 如果目录为空或结构不对，回退到 config 配置
                return _get_default_class_mapping()

            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
            return class_to_idx, idx_to_class, classes
    except Exception as e:
        print(f"Error getting classes from directory: {e}")

    return _get_default_class_mapping()

def _get_default_class_mapping():
    """辅助函数：从 config 返回默认映射"""
    print("Using default classes from config")
    classes = CLASS_NAMES
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
    return class_to_idx, idx_to_class, classes


# ============================================================================
# 2. 权重计算与优化
# ============================================================================

def compute_effective_number_weights(class_counts, beta=0.9999):
    """使用有效数量 (Effective Number) 计算更平衡的类别权重"""
    counts = np.array(list(class_counts.values()), dtype=np.float64)
    counts = np.maximum(counts, 1) # 避免除零

    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    weights = 1.0 / effective_num
    weights = weights / weights.mean() # 归一化

    return weights

def compute_inverse_frequency_weights(class_counts):
    """传统的逆频率加权"""
    counts = np.array(list(class_counts.values()), dtype=np.float64)
    total = counts.sum()
    n_classes = len(counts)
    counts = np.maximum(counts, 1)

    weights = total / (n_classes * counts)
    weights = weights / weights.mean()
    return weights

def apply_confusion_boost(weights, class_names, confusion_classes, boost_factor=1.5):
    """为高混淆类别增加权重"""
    adjusted_weights = weights.copy()
    for i, class_name in enumerate(class_names):
        if class_name in confusion_classes:
            adjusted_weights[i] *= boost_factor
    return adjusted_weights / adjusted_weights.mean()

def get_optimized_class_weights(class_counts, class_names):
    """计算综合优化的类别权重"""
    # 1. 基础权重
    if USE_EFFECTIVE_NUMBER_WEIGHTING:
        weights = compute_effective_number_weights(class_counts, beta=EFFECTIVE_NUMBER_BETA)
        print("Using Effective Number Weighting")
    else:
        weights = compute_inverse_frequency_weights(class_counts)
        print("Using Inverse Frequency Weighting")

    # 2. 针对性增强
    weights = apply_confusion_boost(weights, class_names, HIGH_CONFUSION_CLASSES, CONFUSION_CLASS_BOOST)
    weights = apply_confusion_boost(weights, class_names, LONG_TAIL_CLASSES, LONG_TAIL_BOOST)

    return torch.FloatTensor(weights)


# ============================================================================
# 3. 数据集与采样器
# ============================================================================

class GarbageDataset(Dataset):
    """自定义数据集类，增加了容错处理"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 使用通用函数获取映射
        self.class_to_idx, self.idx_to_class, self.classes = get_class_mapping(self.root_dir)

        self.images = []
        self.labels = []
        self.samples = [] # 兼容 ImageFolder 接口

        # 加载所有图片路径
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            label = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    path = os.path.join(class_dir, img_name)
                    self.images.append(path)
                    self.labels.append(label)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回随机噪音作为替补，防止训练崩溃
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), label

def get_weighted_sampler(dataset, class_names):
    """创建加权采样器"""
    if not USE_WEIGHTED_SAMPLER:
        return None

    sample_weights = []
    for _, label in dataset.samples:
        class_name = class_names[label]
        weight = 1.0

        if class_name in LONG_TAIL_CLASSES:
            weight *= 3.0
        if class_name in HIGH_CONFUSION_CLASSES:
            weight *= 1.5

        sample_weights.append(weight)

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# ============================================================================
# 4. 数据增强与加载器
# ============================================================================

def get_optimized_transforms(image_size=224):
    """获取优化的数据增强变换"""
    resize_size = int(image_size * 1.15)

    # 训练变换
    train_ops = [
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomGrayscale(p=0.1),
    ]

    # 自动增强
    if HAS_AUTOAUGMENT:
        if USE_RANDAUGMENT:
            train_ops.append(RandAugment(num_ops=RANDAUGMENT_N, magnitude=RANDAUGMENT_M))
        elif USE_AUTOAUGMENT:
            train_ops.append(AutoAugment(policy=AutoAugmentPolicy.IMAGENET))

    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])

    train_transform = transforms.Compose(train_ops)

    # 验证变换
    val_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_data_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=0):
    """
    统一的数据加载接口，兼容 train.py 和 app.py
    """
    train_transform, val_transform = get_optimized_transforms(IMAGE_SIZE)

    # 确定目录
    train_dir = os.path.join(data_dir, 'train') if os.path.exists(os.path.join(data_dir, 'train')) else data_dir
    val_dir = os.path.join(data_dir, 'val') if os.path.exists(os.path.join(data_dir, 'val')) else data_dir

    # 创建数据集
    # 优先使用 ImageFolder，如果目录结构复杂则回退到自定义 Dataset
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        # 提取类别信息
        class_to_idx = train_dataset.class_to_idx
        dataset_classes = train_dataset.classes
    except:
        train_dataset = GarbageDataset(train_dir, transform=train_transform)
        val_dataset = GarbageDataset(val_dir, transform=val_transform)
        class_to_idx = train_dataset.class_to_idx
        dataset_classes = train_dataset.classes

    # 准备采样器和权重 (仅供训练使用)
    class_weights = None
    sampler = None

    try:
        # 计算权重 (如果是在训练模式)
        if hasattr(train_dataset, 'targets'):
            targets = train_dataset.targets
        elif hasattr(train_dataset, 'labels'):
            targets = train_dataset.labels
        else:
            targets = [label for _, label in train_dataset.samples]

        class_counts = Counter(targets)
        class_weights = get_optimized_class_weights(class_counts, dataset_classes)

        # 创建采样器
        if USE_WEIGHTED_SAMPLER:
            sampler = get_weighted_sampler(train_dataset, dataset_classes)

    except Exception as e:
        print(f"Note: Skipping weight calculation (inference mode or empty dataset): {e}")

    # 创建加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None), # 如果用了采样器就不能 shuffle
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 返回所有需要的组件
    # 注意：返回值的顺序和数量要适配 train.py 的调用
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return train_loader, val_loader, class_weights, class_to_idx, idx_to_class, dataset_classes


# ============================================================================
# 5. Mixup / CutMix (训练辅助)
# ============================================================================

def mixup_data(x, y, alpha=0.4):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)