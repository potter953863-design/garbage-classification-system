"""
Streamlit Real-time Garbage Classification Application (Ensemble Version)
"""

# Fix OpenMP library conflict on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import requests  # 新增：用于下载模型
import torch
# 开启 Cudnn 自动寻优，对固定尺寸图片(320/384)能加速 10%-20%
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import pandas as pd
import io
import timm  # 关键库: 用于加载 EfficientNetV2
import torch.nn.functional as TorchF

from config import (
    CLASS_NAMES, CLASS_TO_CATEGORY, CATEGORY_COLORS,
    DISTRIBUTION_PLOT_PATH, NUM_CLASSES,
    AVAILABLE_CITIES, get_city_config, get_city_mapping, DATA_DIR,
    PREPARATION_TIPS, ACHIEVEMENTS, BADGE_RARITY_COLORS,
    DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, BASE_DIR
)
from achievement_system import AchievementSystem
from image_enhancement import enhance_image, compare_images, calculate_brightness
from confusion_analysis import check_boundary_warning
from map_service import MapService
from i18n import (
    t,
    get_category_label,
    get_preparation_tip,
    get_achievement_text,
)
import hashlib
import time
import uuid

# ==========================================
# 自动下载模型 (解决 Streamlit Cloud 没模型的问题)
# ==========================================
def download_file(url, save_path):
    """
    检查文件是否存在，不存在则从 URL 下载
    """
    if os.path.exists(save_path):
        return  # 文件已存在，跳过

    # 简单的文件名提取，用于显示
    filename = os.path.basename(save_path)
    st.info(f"正在从 GitHub Release 下载模型文件: {filename}... (首次运行可能需要几分钟)")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # 检查请求是否成功

        # 显示进度条 (可选)
        progress_bar = st.progress(0)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(min(downloaded / total_size, 1.0))

        st.success(f"✅ 下载完成: {filename}")
        time.sleep(1) # 让用户看到成功提示
        st.rerun()    # 重新运行以加载模型

    except Exception as e:
        st.error(f"❌ 模型下载失败: {e}")
        st.error("请检查 app.py 中的下载链接是否正确，或者 GitHub Release 是否已发布。")
        st.stop() # 停止运行



S_MODEL_URL = "sha256:ac3446e1386cd9da78b052673d3541970c768b4237090056bf7918d205d643d1"
M_MODEL_URL = "sha256:79551aa2433cbfa0c298a9f31871aab05ce32d8cf97d5f6668d15626e2ab50bf"

# 执行下载检查 (这会在加载模型前运行)
download_file(S_MODEL_URL, os.path.join(BASE_DIR, 'best_model_fast.pth'))
download_file(M_MODEL_URL, os.path.join(BASE_DIR, 'best_model_sgd.pth'))


# ==========================================
# 模型路径配置
# ==========================================
MODEL_S_PATH = os.path.join(BASE_DIR, 'best_model_fast.pth')
MODEL_M_PATH = os.path.join(BASE_DIR, 'best_model_sgd.pth')
# 如果 SGD 版不存在，尝试加载 Large 版
if not os.path.exists(MODEL_M_PATH):
    MODEL_M_PATH = os.path.join(BASE_DIR, 'best_model_large.pth')

@st.cache_resource
def load_ensemble_models():
    """
    加载双模型 (S版 + M版) 进行融合推理
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}

    # 1. 加载 EfficientNetV2-S (320px)
    if os.path.exists(MODEL_S_PATH):
        try:
            print(f"Loading S-Model from {MODEL_S_PATH}...")
            model_s = timm.create_model('tf_efficientnetv2_s.in1k', num_classes=NUM_CLASSES, pretrained=False)
            checkpoint = torch.load(MODEL_S_PATH, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model_s.load_state_dict(state_dict, strict=False)
            model_s.to(device)
            model_s.eval()
            models['s'] = model_s
        except Exception as e:
            st.error(f"Error loading S-Model: {e}")

    # 2. 加载 EfficientNetV2-M (384px)
    if os.path.exists(MODEL_M_PATH):
        try:
            print(f"Loading M-Model from {MODEL_M_PATH}...")
            model_m = timm.create_model('tf_efficientnetv2_m.in1k', num_classes=NUM_CLASSES, pretrained=False)
            checkpoint = torch.load(MODEL_M_PATH, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model_m.load_state_dict(state_dict, strict=False)
            model_m.to(device)
            model_m.eval()
            models['m'] = model_m
        except Exception as e:
            st.error(f"Error loading M-Model: {e}")

    # 验证模型是否加载成功
    if not models:
        st.error("❌ **CRITICAL: No trained models found!**")
        st.warning(f"Please check if {MODEL_S_PATH} or {MODEL_M_PATH} exists.")

    # 这里我们使用 config 中的 CLASS_NAMES，因为训练时通常保持一致
    # 如果你的 checkpoint 包含 idx_to_class，可以在这里解析，但 timm checkpoint 通常不带这个
    return models, device, CLASS_NAMES

def predict_ensemble(image, models, device, idx_to_class, city_id='default'):
    """
    使用双模型融合进行预测 (加权平均)
    """
    # 图像预处理标准化参数
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # 确保图片是 RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    probs_list = []

    with torch.no_grad():
        # --- 模型 S (权重 0.4) ---
        if 's' in models:
            # S 模型需要 320x320
            transform_s = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
            img_tensor_s = transform_s(image).unsqueeze(0).to(device)
            output_s = models['s'](img_tensor_s)
            prob_s = TorchF.softmax(output_s, dim=1)
            probs_list.append(prob_s * 0.4)

        # --- 模型 M (权重 0.6) ---
        if 'm' in models:
            # M 模型需要 384x384
            transform_m = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
            img_tensor_m = transform_m(image).unsqueeze(0).to(device)
            output_m = models['m'](img_tensor_m)
            prob_m = TorchF.softmax(output_m, dim=1)
            probs_list.append(prob_m * 0.6)

    if not probs_list:
        return None, None, 0.0, np.zeros(NUM_CLASSES)

    # 融合概率
    final_prob_tensor = torch.stack(probs_list).sum(dim=0)
    # 重新归一化 (虽然不严格必要，但为了保险)
    final_prob_tensor = final_prob_tensor / final_prob_tensor.sum()

    # 获取结果
    confidence, predicted_idx = torch.max(final_prob_tensor, 1)
    predicted_idx = predicted_idx.item()
    confidence_score = confidence.item()

    # 获取类别名称
    if predicted_idx < len(idx_to_class):
        predicted_class = idx_to_class[predicted_idx]
    else:
        predicted_class = "Unknown"

    # 获取城市映射类别
    city_mapping = get_city_mapping(city_id)
    predicted_category = city_mapping.get(predicted_class, 'Other')

    # 获取所有概率 (用于分析)
    all_probs = final_prob_tensor.cpu().numpy()[0]

    return predicted_class, predicted_category, confidence_score, all_probs

@st.cache_resource
def get_achievement_system():
    """缓存成就系统实例"""
    return AchievementSystem()

@st.cache_resource
def get_map_service():
    """缓存地图服务实例"""
    return MapService()

def main():
    """
    Main Streamlit application.
    """
    if 'language' not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE

    # Page configuration
    st.set_page_config(
        page_title=t('app.page_title', st.session_state.language),
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    def translate(key: str, **kwargs) -> str:
        return t(key, st.session_state.get('language', DEFAULT_LANGUAGE), **kwargs)

    # Title
    st.title(translate("app.heading"))
    st.markdown(f"**{translate('app.tagline')}**")

    # 加载模型 (Ensemble)
    models, device, idx_to_class = load_ensemble_models()

    # Initialize session state for city selection
    if 'selected_city' not in st.session_state:
        st.session_state.selected_city = 'default'

    # Initialize achievement system (使用缓存)
    achievement_system = get_achievement_system()

    # Initialize user ID in session state (for multi-user support in future)
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 'default'
    if 'last_recorded_signature' not in st.session_state:
        st.session_state.last_recorded_signature = None
    if 'last_recorded_run_id' not in st.session_state:
        st.session_state.last_recorded_run_id = None
    if 'pending_achievement_ids' not in st.session_state:
        st.session_state.pending_achievement_ids = []

    # Sidebar
    with st.sidebar:
        # 显示模型状态
        st.markdown("---")
        if models:
            st.success(f"✅ Ensemble Active ({len(models)} models)")
            st.caption(f"Device: {device}")
            st.caption(f"S-Model: {'Ready' if 's' in models else 'Missing'}")
            st.caption(f"M-Model: {'Ready' if 'm' in models else 'Missing'}")
        else:
            st.error("❌ No Models Loaded")
        st.markdown("---")

        language_options = list(SUPPORTED_LANGUAGES.keys())
        current_language = st.session_state.language
        selected_language = st.selectbox(
            translate("language.selector_label"),
            options=language_options,
            index=language_options.index(current_language),
            format_func=lambda code: SUPPORTED_LANGUAGES.get(code, {}).get("label", code),
            help=translate("language.selector_help"),
            key='language_selector'
        )
        if selected_language != current_language:
            st.session_state.language = selected_language
            st.rerun()

        st.header(translate("sidebar.city_header"))

        # City selector
        city_keys = list(AVAILABLE_CITIES.keys())
        current_index = city_keys.index(st.session_state.selected_city) if st.session_state.selected_city in city_keys else 0

        selected_city = st.selectbox(
            translate("sidebar.city_selector_label"),
            options=city_keys,
            index=current_index,
            format_func=lambda x: AVAILABLE_CITIES[x],
            help=translate("sidebar.city_selector_help"),
            key='city_selector'
        )

        # Update session state
        st.session_state.selected_city = selected_city

        # Get city configuration
        city_config = get_city_config(selected_city)
        city_mapping = get_city_mapping(selected_city)
        city_labels = city_config['category_labels']
        city_colors = city_config['category_colors']

        # Display selected city info
        city_name_cn = city_config.get('name_cn')
        city_extra = f" ({city_name_cn})" if city_name_cn else ""
        st.info(f"📍 **{city_config['name']}**{city_extra}")
        st.caption(translate("sidebar.city_info_caption"))

        st.markdown("---")
        st.header(translate("sidebar.achievements_header"))

        # Get user statistics
        user_stats = achievement_system.get_or_create_user(st.session_state.user_id)
        user_achievements = achievement_system.get_user_achievements(st.session_state.user_id)

        # Display statistics
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric(translate("sidebar.stats.total"), user_stats['total_classifications'])
        with col_stat2:
            st.metric(translate("sidebar.stats.recyclable"), user_stats['recyclable_count'])
        with col_stat3:
            st.metric(translate("sidebar.stats.hazardous"), user_stats['hazardous_count'])
        with col_stat4:
            st.metric(translate("sidebar.stats.kitchen"), user_stats['kitchen_count'])

        # Display unlocked achievements
        if user_achievements:
            st.markdown("### " + translate("sidebar.badges_header"))
            badge_cols = st.columns(min(5, len(user_achievements)))
            for idx, achievement_id in enumerate(user_achievements[:5]):
                if achievement_id in ACHIEVEMENTS:
                    achievement = ACHIEVEMENTS[achievement_id]
                    rarity_color = BADGE_RARITY_COLORS.get(achievement['rarity'], '#9E9E9E')
                    localized_text = get_achievement_text(achievement_id, st.session_state.language)
                    with badge_cols[idx % len(badge_cols)]:
                        badge_html = f"""
                        <div style="
                            text-align: center;
                            padding: 10px;
                            background-color: {rarity_color}20;
                            border: 2px solid {rarity_color};
                            border-radius: 10px;
                            margin: 5px;
                        ">
                            <div style="font-size: 30px;">{achievement['icon']}</div>
                            <div style="font-size: 12px; font-weight: bold; color: {rarity_color};">
                                {localized_text.get('name', achievement['name'])}
                            </div>
                        </div>
                        """
                        st.markdown(badge_html, unsafe_allow_html=True)
                        st.caption(localized_text.get('description', achievement.get('description', '')))

            if len(user_achievements) > 5:
                with st.expander(translate("sidebar.badges_more", count=len(user_achievements))):
                    for achievement_id in user_achievements[5:]:
                        if achievement_id in ACHIEVEMENTS:
                            achievement = ACHIEVEMENTS[achievement_id]
                            rarity_color = BADGE_RARITY_COLORS.get(achievement['rarity'], '#9E9E9E')
                            localized_text = get_achievement_text(achievement_id, st.session_state.language)
                            st.markdown(
                                f"**{achievement['icon']} {localized_text.get('name', achievement['name'])}** - "
                                f"{localized_text.get('description', achievement.get('description', ''))}"
                            )
        else:
            st.info(translate("sidebar.badges_none"))

        st.markdown("---")
        st.header(translate("sidebar.dataset_header"))

        if os.path.exists(DISTRIBUTION_PLOT_PATH):
            st.subheader(translate("sidebar.dataset_chart_title"))
            st.image(DISTRIBUTION_PLOT_PATH)
            st.caption(translate("sidebar.dataset_chart_caption"))
        else:
            st.info(translate("sidebar.dataset_chart_missing"))

        st.markdown("---")
        st.subheader(translate("sidebar.about_header"))
        st.markdown(translate("sidebar.about_text"))

        st.markdown("---")
        st.subheader(translate("sidebar.categories_header", city=city_config['name']))
        for category, color in city_colors.items():
            category_label = city_labels.get(category)
            localized_label = get_category_label(category, st.session_state.language)
            if st.session_state.language == 'zh' and category_label:
                display_label = category_label
            elif category_label and category_label != localized_label and st.session_state.language != 'zh':
                display_label = f"{localized_label} / {category_label}"
            else:
                display_label = localized_label
            st.markdown(
                f"<span style='color: {color}; font-weight: bold;'>●</span> "
                f"**{category}** ({display_label})",
                unsafe_allow_html=True
            )

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(translate("input.section_title"))

        # Image input options
        input_methods = {
            "upload": translate("input.method_upload"),
            "camera": translate("input.method_camera")
        }
        input_option = st.radio(
            translate("input.method_label"),
            options=list(input_methods.keys()),
            format_func=lambda key: input_methods[key],
            horizontal=True
        )

        image = None
        image_signature = None

        if input_option == "upload":
            uploaded_file = st.file_uploader(
                translate("input.upload_label"),
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help=translate("input.upload_help")
            )
            if uploaded_file is not None:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                image = Image.open(io.BytesIO(file_bytes))
                image = image.copy()
                image_signature = hashlib.md5(file_bytes).hexdigest()
        else:
            camera_image = st.camera_input(translate("input.camera_label"), key="camera_input")
            if camera_image is not None:
                image_bytes = camera_image.read()
                image = Image.open(io.BytesIO(image_bytes))
                image = image.copy()
                image_signature = hashlib.md5(image_bytes).hexdigest()

        # Image enhancement
        current_signature = image_signature
        previous_signature = st.session_state.get("active_image_signature")
        if current_signature:
            if previous_signature != current_signature:
                st.session_state["active_image_signature"] = current_signature
                st.session_state.pop("last_prediction", None)
                st.session_state["last_recorded_signature"] = None
                st.session_state["last_recorded_run_id"] = None
        else:
            if previous_signature:
                st.session_state.pop("active_image_signature", None)
                st.session_state.pop("last_prediction", None)
                st.session_state["last_recorded_signature"] = None
                st.session_state["last_recorded_run_id"] = None

        if image is not None:
            st.markdown(translate("enhancement.section_title"))
            use_enhancement = st.checkbox(
                translate("enhancement.toggle_label"),
                value=True,
                help=translate("enhancement.toggle_help")
            )

            if use_enhancement:
                enhanced_image, enhancement_info = enhance_image(
                    image,
                    auto_brightness=True,
                    denoise=True,
                    sharpen=True
                )

                original_brightness = enhancement_info['original_brightness']
                brightness_status = (
                    translate("enhancement.metric.adjusted")
                    if enhancement_info['brightness_adjusted']
                    else translate("enhancement.metric.normal")
                )

                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric(translate("enhancement.metric.original"), f"{original_brightness*100:.1f}%",
                              delta=f"{brightness_status}")
                with col_info2:
                    st.metric(
                        translate("enhancement.metric.denoise"),
                        translate("enhancement.metric.applied") if enhancement_info['denoised'] else translate("enhancement.metric.skipped")
                    )
                with col_info3:
                    st.metric(
                        translate("enhancement.metric.sharpen"),
                        translate("enhancement.metric.applied") if enhancement_info['sharpened'] else translate("enhancement.metric.skipped")
                    )

                with st.expander(translate("enhancement.comparison_title")):
                    comparison = compare_images(image, enhanced_image)
                    st.image(comparison, caption=translate("enhancement.comparison_caption"))

                image = enhanced_image
            else:
                original_brightness = calculate_brightness(image)
                if original_brightness < 0.3:
                    st.warning(translate("enhancement.dark_warning", value=original_brightness*100))

        if image is not None:
            st.image(image, caption=translate("image.caption_ready"))

    with col2:
        st.subheader(translate("results.section_title"))
        stored_result = st.session_state.get("last_prediction")
        result_data = None
        has_image = image is not None
        classify_clicked = False

        if has_image:
            classify_clicked = st.button(
                translate("results.start_button"),
                key="start_classification",
                type="primary",
                use_container_width=True,
                help=translate("results.start_help")
            )

        if classify_clicked and has_image:
            classification_run_id = f"{st.session_state.get('active_image_signature') or 'no_sig'}-{uuid.uuid4().hex}"

            # 使用双模型融合进行预测
            current_city = st.session_state.get('selected_city', 'default')

            with st.spinner(translate("results.spinner_classify")):
                predicted_class, predicted_category, confidence, all_probs = predict_ensemble(
                    image, models, device, idx_to_class, city_id=current_city
                )

            # 混淆警告检测
            has_warning, warning_msg, confused_class = check_boundary_warning(
                predicted_class,
                all_probs,
                idx_to_class, # 使用类别列表
                confidence_threshold=0.5, # Ensemble 更自信，可以提高阈值
                diff_threshold=0.15
            )

            boundary_warning = None
            if has_warning:
                boundary_warning = {
                    'tip': warning_msg,
                    'confused_class': confused_class
                }

            idx_mapping = idx_to_class
            dataset_list = idx_to_class

            result_data = {
                "image_signature": st.session_state.get("active_image_signature"),
                "city_id": current_city,
                "predicted_class": predicted_class,
                "predicted_category": predicted_category,
                "confidence": float(confidence),
                "all_probs": all_probs.tolist(),
                "boundary_warning": boundary_warning,
                "idx_to_class": idx_mapping,
                "dataset_classes": dataset_list,
                "timestamp": time.time(),
                "classification_run_id": classification_run_id
            }
            st.session_state["last_prediction"] = result_data

            should_record = predicted_class and confidence > 0.1
            if should_record and st.session_state.get("last_recorded_run_id") == classification_run_id:
                should_record = False

            if should_record:
                achievement_system.record_classification(
                    user_id=st.session_state.user_id,
                    class_name=predicted_class,
                    category=predicted_category,
                    confidence=confidence,
                    user_id_param=st.session_state.user_id
                )

                newly_unlocked = achievement_system.check_and_unlock_achievements(
                    user_id=st.session_state.user_id,
                    achievement_config=ACHIEVEMENTS
                )

                if newly_unlocked:
                    st.session_state["pending_achievement_ids"] = newly_unlocked

                st.session_state["last_recorded_signature"] = st.session_state.get("active_image_signature")
                st.session_state["last_recorded_run_id"] = classification_run_id
                st.session_state["stats_refresh_trigger"] = time.time()
                st.rerun()
        elif stored_result:
            result_data = stored_result

        if result_data:
            result_city = result_data.get("city_id", st.session_state.get('selected_city', 'default'))
            city_config = get_city_config(result_city)
            city_mapping = get_city_mapping(result_city)
            city_labels = city_config['category_labels']
            city_colors = city_config['category_colors']

            predicted_class = result_data['predicted_class']
            predicted_category = result_data['predicted_category']
            confidence = result_data['confidence']
            all_probs = np.array(result_data['all_probs'])
            boundary_warning = result_data.get('boundary_warning')
            stored_idx_to_class = result_data.get('idx_to_class')

            def resolve_class_name_from_idx(class_idx: int) -> str:
                if isinstance(stored_idx_to_class, list):
                    if 0 <= class_idx < len(stored_idx_to_class):
                        return stored_idx_to_class[class_idx]
                return f"Class_{class_idx}"

            city_name_display = city_config['name']
            if city_config.get('name_cn'):
                city_name_display = f"{city_config['name']} ({city_config['name_cn']})"
            st.caption(translate("results.city_caption", city_name=city_name_display))
            st.markdown(translate("results.result_heading"))

            category_color = city_colors.get(predicted_category, '#808080')
            city_label_value = city_labels.get(predicted_category)
            localized_category_label = get_category_label(predicted_category, st.session_state.language)
            if st.session_state.language == 'zh' and city_label_value:
                category_label_display = city_label_value
            elif city_label_value and city_label_value != localized_category_label and st.session_state.language != 'zh':
                category_label_display = f"{localized_category_label} / {city_label_value}"
            else:
                category_label_display = localized_category_label

            category_line = translate("results.category_label", category=predicted_category, label=category_label_display)
            confidence_line = translate("results.confidence_label", value=confidence * 100)

            result_html = f"""
            <div style="
                background-color: {category_color}20;
                border-left: 5px solid {category_color};
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <h3 style="color: {category_color}; margin: 0;">
                    {predicted_class.replace('-', ' ').title()}
                </h3>
                <p style="margin: 5px 0; font-size: 18px;">
                    {category_line}
                </p>
                <p style="margin: 5px 0; font-size: 16px;">
                    {confidence_line}
                </p>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)

            pending_achievements = st.session_state.pop("pending_achievement_ids", None)
            if pending_achievements:
                st.markdown("---")
                st.markdown(translate("achievements.new_title"))
                for achievement_id in pending_achievements:
                    if achievement_id in ACHIEVEMENTS:
                        achievement = ACHIEVEMENTS[achievement_id]
                        rarity_color = BADGE_RARITY_COLORS.get(achievement['rarity'], '#9E9E9E')
                        localized_text = get_achievement_text(achievement_id, st.session_state.language)
                        unlock_html = f"""
                        <div style="
                            background: linear-gradient(135deg, {rarity_color}20 0%, {rarity_color}40 100%);
                            border: 3px solid {rarity_color};
                            padding: 20px;
                            border-radius: 15px;
                            margin: 10px 0;
                            text-align: center;
                        ">
                            <div style="font-size: 50px; margin-bottom: 10px;">
                                {achievement['icon']}
                            </div>
                            <h3 style="color: {rarity_color}; margin: 5px 0;">
                                {localized_text.get('name', achievement['name'])}
                            </h3>
                            <p style="margin: 5px 0; font-size: 14px;">
                                {localized_text.get('description', achievement.get('description', ''))}
                            </p>
                        </div>
                        """
                        st.markdown(unlock_html, unsafe_allow_html=True)
                st.balloons()

            localized_tip = get_preparation_tip(predicted_class, st.session_state.language)
            if localized_tip:
                st.markdown("---")
                st.markdown(translate("tips.section_title"))
                tip_content = translate(
                    "tips.card_prefix",
                    class_name=predicted_class.replace('-', ' ').title(),
                    tip=localized_tip
                )
                tip_html = f"""
                <div style="
                    background-color: #E3F2FD;
                    border-left: 4px solid #2196F3;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                ">
                    <p style="margin: 0; font-size: 15px; color: #1565C0;">
                        {tip_content}
                    </p>
                </div>
                """
                st.markdown(tip_html, unsafe_allow_html=True)

            if boundary_warning:
                st.markdown("---")
                st.markdown(translate("warning.section_title"))
                st.warning(boundary_warning['tip'], icon="⚠️")

            st.progress(confidence)

            st.markdown(translate("top.section_title"))
            top3_indices = np.argsort(all_probs)[-3:][::-1]

            for i, idx in enumerate(top3_indices):
                class_name = resolve_class_name_from_idx(idx)
                prob = all_probs[idx]
                category = city_mapping.get(class_name, "Unknown")
                color = city_colors.get(category, "#607D8B")

                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(
                        f"**{i+1}. {class_name.replace('-', ' ').title()}** "
                        f"({category})"
                    )
                with col_b:
                    st.markdown(f"**{prob*100:.1f}%**")

                st.markdown(
                    f'<div style="background-color: {color}40; height: 8px; '
                    f'width: {prob*100}%; border-radius: 4px; margin-bottom: 10px;"></div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            popup_key = f"recycling_popup_{predicted_category}"
            user_choice_key = f"user_choice_{predicted_category}"

            if user_choice_key not in st.session_state:
                st.info(
                    f"{translate('location.prompt_title', label=category_label_display)}\n\n"
                    f"{translate('location.prompt_body')}"
                )

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button(translate("location.prompt_yes"), key=f"yes_{popup_key}", type="primary"):
                        st.session_state[user_choice_key] = True
                        st.rerun()
                with col2:
                    if st.button(translate("location.prompt_no"), key=f"no_{popup_key}"):
                        st.session_state[user_choice_key] = False
                        st.rerun()
                with col3:
                    if st.button(translate("location.prompt_later"), key=f"later_{popup_key}"):
                        st.session_state[user_choice_key] = None
                        st.rerun()

            show_location_finder = st.session_state.get(user_choice_key, False)

            if show_location_finder:
                # 使用缓存的地图服务
                map_service = get_map_service()
                default_location_label = translate("location.default_location")
                
                st.markdown(translate("location.input_title"))
                st.info(translate("location.input_hint"))
                
                user_lat = None
                user_lon = None
                start_label = None
                coordinates_ready = False
                address_input = None
                address_text_for_query = None
                
                location_methods = {
                    "address": translate("location.method_address"),
                    "coordinates": translate("location.method_coordinates")
                }
                location_method = st.radio(
                    translate("location.method_label"),
                    list(location_methods.keys()),
                    format_func=lambda key: location_methods[key],
                    horizontal=True,
                    key=f"location_method_{predicted_category}"
                )
                
                if location_method == "address":
                    address_input = st.text_input(
                        translate("location.address_label"),
                        key=f"address_input_{predicted_category}",
                        placeholder=translate("location.address_placeholder")
                    )
                    
                    if address_input:
                        with st.spinner(translate("location.address_spinner")):
                            coords = map_service.geocode(address_input)
                            if coords:
                                user_lat, user_lon = coords
                                cleaned_label = address_input.strip()
                                start_label = cleaned_label if cleaned_label else default_location_label
                                address_text_for_query = start_label
                                coordinates_ready = True
                                st.success(translate("location.address_success", lat=user_lat, lon=user_lon))
                            else:
                                st.warning(translate("location.address_fail"))
                                coordinates_ready = False
                else:
                    col_lat, col_lon = st.columns(2)
                    with col_lat:
                        user_lat = st.number_input(
                            translate("location.coords_lat"),
                            value=31.2304,
                            format="%.6f",
                            key=f"lat_input_{predicted_category}"
                        )
                    with col_lon:
                        user_lon = st.number_input(
                            translate("location.coords_lon"),
                            value=121.4737,
                            format="%.6f",
                            key=f"lon_input_{predicted_category}"
                        )
                    
                    start_label_input = st.text_input(
                        translate("location.coords_name_label"),
                        key=f"start_label_{predicted_category}",
                        placeholder=translate("location.coords_name_placeholder")
                    )
                    
                    if user_lat and user_lon:
                        coordinates_ready = True
                        start_label = start_label_input.strip() if start_label_input else default_location_label
                        address_text_for_query = start_label
                
                
                search_triggered = False
                if coordinates_ready and user_lat and user_lon:
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        search_triggered = st.button(
                            translate("location.search_button"),
                            key=f"search_button_{predicted_category}",
                            type="primary",
                            use_container_width=True
                        )
                
                if search_triggered and user_lat and user_lon:
                    try:
                        with st.spinner(translate("location.search_spinner")):
                            poi_search_url = map_service.generate_poi_search_url(
                                category=predicted_category,
                                language=st.session_state.language,
                                latitude=user_lat,
                                longitude=user_lon,
                                address_text=address_text_for_query
                            )
                        st.success(translate("location.search_success"))
                        st.markdown("---")
                        link_text = translate("location.search_link_text", label=category_label_display)
                        st.markdown(
                            f"[{link_text}]({poi_search_url})"
                        )
                        st.caption(translate("location.search_caption"))
                    except Exception as e:
                        st.error(translate("location.search_error", error=e))
                else:
                    st.info(translate("location.search_prompt"))
            
            with st.expander(translate("results.all_probs_title")):
                prob_data = []
                classes_to_use = stored_idx_to_class if stored_idx_to_class is not None else CLASS_NAMES
                for idx in range(len(classes_to_use)):
                    class_name = resolve_class_name_from_idx(idx)
                    prob = all_probs[idx] if idx < len(all_probs) else 0.0
                    category = city_mapping.get(class_name, "Unknown")
                    category_label = city_labels.get(category, "Unknown")
                    prob_data.append({
                        "Class": class_name.replace('-', ' ').title(),
                        "Category": f"{category} ({category_label})",
                        "Probability": f"{prob*100:.2f}%"
                    })
                
                df = pd.DataFrame(prob_data)
                df = df.sort_values("Probability", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
                st.dataframe(df, hide_index=True)
        else:
            if has_image:
                st.info(translate("results.need_click"))
            else:
                st.info(translate("results.need_image"))
                st.markdown(translate("results.howto_title"))
                st.markdown(translate("results.howto_upload"))
                st.markdown(translate("results.howto_camera"))
                st.markdown(translate("results.howto_button"))
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        f"{translate('footer.text')}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()