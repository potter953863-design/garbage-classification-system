"""
Model definition for Garbage Classification System
Updated to match the training architecture (EfficientNetV2-S via timm)
"""

# Fix OpenMP library conflict on Windows
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import timm  # 必须确保安装了 timm (pip install timm)
from config import NUM_CLASSES


def GarbageClassifier(num_classes=NUM_CLASSES, pretrained=False, dropout=0.0, model_name='efficientnetv2_s'):
    """
    Factory function to create the model.
    Using 'timm' to match the architecture used in train.py.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (usually False for inference)
        dropout: Dropout rate (not always used by all timm models during inference)
        model_name: The name of the model architecture

    Returns:
        The initialized model
    """
    # 你的训练日志显示使用的是 'EfficientNetV2-S'
    # 在 timm 中，标准名称通常是 'efficientnetv2_s' 或 'tf_efficientnetv2_s'

    target_model_name = 'efficientnetv2_s'

    try:
        # 尝试创建模型
        model = timm.create_model(
            target_model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    except Exception as e:
        print(f"Loading {target_model_name} failed, trying 'tf_efficientnetv2_s': {e}")
        # 备用名称，有时训练脚本使用的是 tf_ 前缀的版本
        target_model_name = 'tf_efficientnetv2_s'
        model = timm.create_model(
            target_model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    return model


# 保持 create_model 函数以便兼容 app.py 中的调用习惯
def create_model(num_classes=NUM_CLASSES, pretrained=True, device='cpu', dropout=0.2, model_name=None):
    """
    Factory function wrapper.
    """
    model = GarbageClassifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    model = model.to(device)

    # 打印简要信息
    print(f"\nModel Summary:")
    print(f"Architecture: EfficientNetV2-S (timm)")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {device}")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = create_model(device=device)

        # Test forward pass
        dummy_input = torch.randn(1, 3, 320, 320).to(device)  # 训练时使用的是 320x320
        output = model(dummy_input)

        print(f"\nTest forward pass:")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output logits: {output[0][:5]}...")
        print("\n✅ Model created and tested successfully!")

        # 检查是否包含报错中缺失的层，例如 conv_stem
        # 如果能打印出来，说明结构是对的
        print("\nChecking first layer (conv_stem):")
        if hasattr(model, 'conv_stem'):
            print(model.conv_stem)
        else:
            print("Note: Structure differs from standard timm efficientnet, please verify.")

    except ImportError:
        print("❌ Error: 'timm' library is missing. Please install it using: pip install timm")
    except Exception as e:
        print(f"❌ Error during model creation: {e}")