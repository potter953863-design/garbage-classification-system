"""
Configuration file for Garbage Classification System (Deployment Version)
仅包含 App 运行所需的配置，移除了冗余训练参数。
"""

import os
import json

from i18n import (
    SUPPORTED_LANGUAGES as I18N_SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE as I18N_DEFAULT_LANGUAGE,
    CATEGORY_LABELS,
    PREPARATION_TIPS_I18N,
)

SUPPORTED_LANGUAGES = I18N_SUPPORTED_LANGUAGES
DEFAULT_LANGUAGE = I18N_DEFAULT_LANGUAGE

# =============================================================================
# Dataset paths (精简版 - 移除 trainval 依赖)
# =============================================================================
# 获取当前文件所在目录的绝对路径，确保云端/本地部署时路径正确
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 关键修改：直接指向根目录下的文件 (假设你已将文件从 trainval 移出)
TRAIN_CLASSES_PATH = os.path.join(BASE_DIR, "train_classes.txt")
CLASSIFY_RULE_PATH = os.path.join(BASE_DIR, "classify_rule.json")

# 模型文件路径
# 注意：这里只定义路径变量，具体加载逻辑由 app.py 处理
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'best_model_large.pth') # 默认占位，实际 App 会加载 fast 和 sgd

# 资源路径
DISTRIBUTION_PLOT_PATH = os.path.join(BASE_DIR, 'distribution.png')
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, 'confusion_matrix.png')
CONFUSION_MATRIX_DATA_PATH = os.path.join(BASE_DIR, 'confusion_matrix_data.npz')
DATA_DIR = os.path.join(BASE_DIR, "datasets", "cropped_train") # 仅作路径引用，不强制要求存在

def _load_train_classes(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []

def _load_classify_rule(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

TRAIN_CLASSES = _load_train_classes(TRAIN_CLASSES_PATH)
CLASSIFY_RULE = _load_classify_rule(CLASSIFY_RULE_PATH)

_CAT_MAP = {
    "可回收物": "Recyclable",
    "厨余垃圾": "Kitchen",
    "有害垃圾": "Hazardous",
    "其他垃圾": "Other",
}

CLASS_TO_CATEGORY = {}
for cat_cn, names in CLASSIFY_RULE.items():
    mapped_cat = _CAT_MAP.get(cat_cn, "Other")
    for n in names:
        CLASS_TO_CATEGORY[n] = mapped_cat

# 如果没加载到文件，使用默认备用列表
if not CLASS_TO_CATEGORY:
    CLASS_TO_CATEGORY = {
        'paper': 'Recyclable', 'cardboard': 'Recyclable',
        'brown-glass': 'Recyclable', 'green-glass': 'Recyclable',
        'white-glass': 'Recyclable', 'metal': 'Recyclable',
        'plastic': 'Recyclable', 'clothes': 'Recyclable',
        'shoes': 'Recyclable', 'battery': 'Hazardous',
        'biological': 'Kitchen', 'trash': 'Other'
    }

CATEGORY_COLORS = {
    'Recyclable': '#1E88E5',
    'Hazardous': '#D32F2F',
    'Kitchen': '#388E3C',
    'Other': '#616161'
}

CATEGORY_LABELS_CN = CATEGORY_LABELS.get('zh', {})

if TRAIN_CLASSES:
    CLASS_NAMES = TRAIN_CLASSES
else:
    CLASS_NAMES = sorted(CLASS_TO_CATEGORY.keys())

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# Map Service Configuration
# ============================================================================
GAODE_API_KEY = os.environ.get('GAODE_API_KEY', '8a0877a90a140a273601590e549527da')
GAODE_API_BASE_URL = 'https://restapi.amap.com/v3'
DEFAULT_SEARCH_RADIUS = 5.0

# ============================================================================
# Preparation Tips
# ============================================================================
PREPARATION_TIPS = PREPARATION_TIPS_I18N.get('zh', {})

# ============================================================================
# Achievement System Configuration
# ============================================================================
ACHIEVEMENTS = {
    'first_classification': {
        'name': '垃圾分类新手', 'name_en': 'Classification Beginner',
        'description': '完成第一次垃圾分类', 'description_en': 'Complete your first classification',
        'icon': '🌱', 'type': 'total_classifications', 'threshold': 1, 'rarity': 'common'
    },
    'ten_classifications': {
        'name': '分类小能手', 'name_en': 'Classification Apprentice',
        'description': '完成10次垃圾分类', 'description_en': 'Complete 10 classifications',
        'icon': '⭐', 'type': 'total_classifications', 'threshold': 10, 'rarity': 'common'
    },
    'fifty_classifications': {
        'name': '分类达人', 'name_en': 'Classification Expert',
        'description': '完成50次垃圾分类', 'description_en': 'Complete 50 classifications',
        'icon': '🏆', 'type': 'total_classifications', 'threshold': 50, 'rarity': 'rare'
    },
    'hundred_classifications': {
        'name': '分类大师', 'name_en': 'Classification Master',
        'description': '完成100次垃圾分类', 'description_en': 'Complete 100 classifications',
        'icon': '👑', 'type': 'total_classifications', 'threshold': 100, 'rarity': 'epic'
    },
    'five_hundred_classifications': {
        'name': '环保传奇', 'name_en': 'Environmental Legend',
        'description': '完成500次垃圾分类', 'description_en': 'Complete 500 classifications',
        'icon': '🌟', 'type': 'total_classifications', 'threshold': 500, 'rarity': 'legendary'
    },
    'recyclable_enthusiast': {
        'name': '回收达人', 'name_en': 'Recyclable Enthusiast',
        'description': '识别50件可回收物', 'description_en': 'Classify 50 recyclable items',
        'icon': '♻️', 'type': 'category_count', 'category': 'Recyclable', 'threshold': 50, 'rarity': 'rare'
    },
    'hazardous_guardian': {
        'name': '有害垃圾守护者', 'name_en': 'Hazardous Guardian',
        'description': '识别10件有害垃圾', 'description_en': 'Classify 10 hazardous items',
        'icon': '⚠️', 'type': 'hazardous_count', 'threshold': 10, 'rarity': 'epic'
    },
    'hazardous_expert': {
        'name': '有害垃圾专家', 'name_en': 'Hazardous Expert',
        'description': '识别50件有害垃圾', 'description_en': 'Classify 50 hazardous items',
        'icon': '🛡️', 'type': 'hazardous_count', 'threshold': 50, 'rarity': 'legendary'
    },
    'kitchen_warrior': {
        'name': '厨余战士', 'name_en': 'Kitchen Warrior',
        'description': '识别30件厨余垃圾', 'description_en': 'Classify 30 kitchen waste items',
        'icon': '🍃', 'type': 'category_count', 'category': 'Kitchen', 'threshold': 30, 'rarity': 'rare'
    },
    'all_rounder': {
        'name': '全能分类师', 'name_en': 'All-Round Classifier',
        'description': '识别过所有4大类别的垃圾', 'description_en': 'Classify items from all 4 categories',
        'icon': '🎯', 'type': 'all_categories', 'threshold': 1, 'rarity': 'epic'
    }
}

BADGE_RARITY_COLORS = {
    'common': '#9E9E9E',
    'rare': '#2196F3',
    'epic': '#9C27B0',
    'legendary': '#FF9800'
}

# ============================================================================
# City Configuration
# ============================================================================

DEFAULT_CITY_CONFIG = {
    'name': 'Default (National Standard)',
    'name_cn': '默认（国家标准）',
    'class_to_category': CLASS_TO_CATEGORY.copy(),
    'category_labels': CATEGORY_LABELS_CN.copy(),
    'category_colors': CATEGORY_COLORS.copy()
}

SHANGHAI_CONFIG = {
    'name': 'Shanghai', 'name_cn': '上海',
    'class_to_category': CLASS_TO_CATEGORY.copy(),
    'category_labels': {'Recyclable': '可回收物', 'Hazardous': '有害垃圾', 'Kitchen': '湿垃圾', 'Other': '干垃圾'},
    'category_colors': CATEGORY_COLORS.copy()
}

BEIJING_CONFIG = {
    'name': 'Beijing', 'name_cn': '北京',
    'class_to_category': CLASS_TO_CATEGORY.copy(),
    'category_labels': {'Recyclable': '可回收物', 'Hazardous': '有害垃圾', 'Kitchen': '厨余垃圾', 'Other': '其他垃圾'},
    'category_colors': CATEGORY_COLORS.copy()
}

SHENZHEN_CONFIG = {
    'name': 'Shenzhen', 'name_cn': '深圳',
    'class_to_category': CLASS_TO_CATEGORY.copy(),
    'category_labels': {'Recyclable': '可回收物', 'Hazardous': '有害垃圾', 'Kitchen': '易腐垃圾', 'Other': '其他垃圾'},
    'category_colors': CATEGORY_COLORS.copy()
}

GUANGZHOU_CONFIG = {
    'name': 'Guangzhou', 'name_cn': '广州',
    'class_to_category': CLASS_TO_CATEGORY.copy(),
    'category_labels': {'Recyclable': '可回收物', 'Hazardous': '有害垃圾', 'Kitchen': '餐厨垃圾', 'Other': '其他垃圾'},
    'category_colors': CATEGORY_COLORS.copy()
}

CITY_CONFIGS = {
    'default': DEFAULT_CITY_CONFIG,
    'shanghai': SHANGHAI_CONFIG,
    'beijing': BEIJING_CONFIG,
    'shenzhen': SHENZHEN_CONFIG,
    'guangzhou': GUANGZHOU_CONFIG
}

AVAILABLE_CITIES = {
    'default': 'Default (National Standard) / 默认（国家标准）',
    'shanghai': 'Shanghai / 上海',
    'beijing': 'Beijing / 北京',
    'shenzhen': 'Shenzhen / 深圳',
    'guangzhou': 'Guangzhou / 广州'
}

def get_city_config(city_id='default'):
    return CITY_CONFIGS.get(city_id, DEFAULT_CITY_CONFIG)

def get_city_mapping(city_id='default'):
    config = get_city_config(city_id)
    return config['class_to_category']