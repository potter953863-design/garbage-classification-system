"""
Confusion Analysis Module
Provides boundary warning functionality for similar classes
"""

import numpy as np
import os
from config import CONFUSION_MATRIX_DATA_PATH


def load_confusion_matrix():
    """
    Load confusion matrix data from file.
    
    Returns:
        tuple: (confusion_matrix, class_names) or (None, None) if not found
    """
    if not os.path.exists(CONFUSION_MATRIX_DATA_PATH):
        return None, None
    
    try:
        data = np.load(CONFUSION_MATRIX_DATA_PATH, allow_pickle=True)
        confusion_matrix = data['confusion_matrix']
        class_names = data['class_names']
        return confusion_matrix, class_names
    except Exception as e:
        print(f"Error loading confusion matrix: {e}")
        return None, None


def check_boundary_warning(predicted_class, all_probs, class_names, 
                          confusion_matrix=None, 
                          threshold=0.15,
                          confidence_threshold=None,
                          diff_threshold=None,
                          **kwargs):
    """
    Check if prediction is near decision boundary and provide warning.
    
    Args:
        predicted_class: Predicted class name
        all_probs: All class probabilities (numpy array)
        class_names: List of all class names
        confusion_matrix: Optional confusion matrix for better analysis
        threshold: Probability difference threshold for boundary warning
        confidence_threshold: Alternative name for threshold
        diff_threshold: Another alternative name for threshold
        **kwargs: Accept any other parameters for compatibility
        
    Returns:
        tuple: (has_warning: bool, warning_message: str, confused_class: str or None)
    """
    # 优先级: diff_threshold > confidence_threshold > threshold
    if diff_threshold is not None:
        actual_threshold = diff_threshold
    elif confidence_threshold is not None:
        actual_threshold = confidence_threshold
    else:
        actual_threshold = threshold
    
    # 检查是否有足够的类别
    if len(all_probs) < 2:
        # 只有1个类别，无法进行边界检查
        return False, None, None
    
    try:
        # Get top 2 predictions
        # 确保 all_probs 是数值类型，防止字符串排序错误
        # 如果这里传入了字符串列表，argsort 可能产生意外结果，但下面的 float 转换会拦截错误
        top2_indices = np.argsort(all_probs)[-2:][::-1]
        
        # --- 修复部分开始: 安全获取概率并强制转换为 float ---
        # 这可以防止 'str' - 'float' 的 TypeError
        top1_prob = float(all_probs[top2_indices[0]])
        top2_prob = float(all_probs[top2_indices[1]]) if len(top2_indices) > 1 else 0.0
        
        # Check if close to boundary
        prob_diff = top1_prob - top2_prob
        # --- 修复部分结束 ---

        if prob_diff < actual_threshold and len(top2_indices) > 1:
            # Close to decision boundary
            confused_class = class_names[top2_indices[1]]
            
            warning_msg = (
                f"⚠️ **边界警告**: 模型在 **{predicted_class}** 和 **{confused_class}** "
                f"之间不确定 (概率差: {prob_diff*100:.1f}%)\n\n"
                f"**建议**: \n"
                f"- 检查物品的材质和特征\n"
                f"- 如有疑问,请参考详细分类指南\n"
                f"- {predicted_class}: {top1_prob*100:.1f}%\n"
                f"- {confused_class}: {top2_prob*100:.1f}%"
            )
            
            return True, warning_msg, confused_class

    except (ValueError, TypeError, IndexError) as e:
        # 捕获类型转换错误或索引错误，防止程序崩溃
        print(f"Warning check failed due to data error: {e}")
        return False, None, None
    
    return False, None, None


def get_confusion_pairs(confusion_matrix, class_names, top_k=10):
    """
    Get top K most confused class pairs from confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix (numpy array)
        class_names: List of class names
        top_k: Number of top confused pairs to return
        
    Returns:
        list: List of tuples (class1, class2, confusion_count)
    """
    if confusion_matrix is None:
        return []
    
    confusion_pairs = []
    n_classes = len(class_names)
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and confusion_matrix[i, j] > 0:
                confusion_pairs.append((
                    class_names[i],
                    class_names[j],
                    int(confusion_matrix[i, j])
                ))
    
    # Sort by confusion count
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return confusion_pairs[:top_k]


def analyze_prediction_confidence(all_probs, class_names, predicted_class):
    """
    Analyze prediction confidence and provide insights.
    
    Args:
        all_probs: All class probabilities
        class_names: List of class names
        predicted_class: Predicted class name
        
    Returns:
        dict: Analysis results including confidence level and alternatives
    """
    if len(all_probs) == 0:
        return {
            'confidence_level': 'unknown',
            'confidence_text': '未知',
            'confidence_score': 0.0,
            'alternatives': [],
            'entropy': 0.0
        }
    
    try:
        top5_indices = np.argsort(all_probs)[-min(5, len(all_probs)):][::-1]
        
        # Confidence level
        # 同样确保这里的 max_prob 是 float
        max_prob = float(np.max(all_probs))
        
        if max_prob > 0.9:
            confidence_level = "very_high"
            confidence_text = "非常高"
        elif max_prob > 0.7:
            confidence_level = "high"
            confidence_text = "高"
        elif max_prob > 0.5:
            confidence_level = "medium"
            confidence_text = "中等"
        elif max_prob > 0.3:
            confidence_level = "low"
            confidence_text = "低"
        else:
            confidence_level = "very_low"
            confidence_text = "很低"
        
        # Alternative predictions
        alternatives = []
        for idx in top5_indices[1:]:  # Skip the top prediction
            if idx < len(class_names):
                alternatives.append({
                    'class': class_names[idx],
                    'probability': float(all_probs[idx])
                })
        
        # 安全计算熵
        # 确保 all_probs 是 float 类型的 numpy 数组
        probs_safe = np.array(all_probs, dtype=float)
        probs_safe = np.clip(probs_safe, 1e-10, 1.0)
        entropy = float(-np.sum(probs_safe * np.log(probs_safe)))
        
        return {
            'confidence_level': confidence_level,
            'confidence_text': confidence_text,
            'confidence_score': max_prob,
            'alternatives': alternatives,
            'entropy': entropy
        }
    except Exception as e:
        # 如果分析出错，返回默认安全值
        print(f"Confidence analysis failed: {e}")
        return {
            'confidence_level': 'unknown',
            'confidence_text': '计算错误',
            'confidence_score': 0.0,
            'alternatives': [],
            'entropy': 0.0
        }