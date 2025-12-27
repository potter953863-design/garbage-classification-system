"""
Image Enhancement Module for Garbage Classification
Provides automatic brightness/contrast adjustment, denoising, and sharpening
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional


def calculate_brightness(image: Image.Image) -> float:
    """
    Calculate the average brightness of an image.
    
    Args:
        image: PIL Image
        
    Returns:
        Average brightness value (0.0 to 1.0)
    """
    # Convert to grayscale
    gray = image.convert('L')
    # Calculate mean brightness
    brightness = np.mean(np.array(gray)) / 255.0
    return brightness


def auto_adjust_brightness_contrast(image: Image.Image, 
                                   target_brightness: float = 0.5,
                                   brightness_threshold: float = 0.3) -> Image.Image:
    """
    Automatically adjust brightness and contrast if image is too dark.
    
    Args:
        image: PIL Image
        target_brightness: Target brightness level (0.0 to 1.0)
        brightness_threshold: Minimum brightness threshold to trigger adjustment
        
    Returns:
        Enhanced PIL Image
    """
    # Calculate current brightness
    current_brightness = calculate_brightness(image)
    
    # Only adjust if image is too dark
    if current_brightness < brightness_threshold:
        # Calculate brightness factor
        brightness_factor = target_brightness / current_brightness
        # Clamp to reasonable range (0.5 to 2.0)
        brightness_factor = max(0.5, min(2.0, brightness_factor))
        
        # Apply brightness enhancement
        enhancer = ImageEnhance.Brightness(image)
        enhanced = enhancer.enhance(brightness_factor)
        
        # Also slightly increase contrast for better visibility
        contrast_factor = 1.0 + (brightness_threshold - current_brightness) * 0.5
        contrast_factor = max(1.0, min(1.3, contrast_factor))
        
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(contrast_factor)
        
        return enhanced
    
    return image


def denoise_image(image: Image.Image, method: str = 'median') -> Image.Image:
    """
    Apply denoising to reduce image noise.
    
    Args:
        image: PIL Image
        method: Denoising method ('median' or 'gaussian')
        
    Returns:
        Denoised PIL Image
    """
    if method == 'median':
        # Median filter - good for salt-and-pepper noise
        return image.filter(ImageFilter.MedianFilter(size=3))
    elif method == 'gaussian':
        # Gaussian blur - good for general noise reduction
        return image.filter(ImageFilter.GaussianBlur(radius=0.5))
    else:
        return image


def sharpen_image(image: Image.Image, factor: float = 1.2) -> Image.Image:
    """
    Sharpen image to enhance edges and details.
    
    Args:
        image: PIL Image
        factor: Sharpening factor (1.0 = no change, >1.0 = sharper)
        
    Returns:
        Sharpened PIL Image
    """
    enhancer = ImageEnhance.Sharpness(image)
    # Clamp factor to reasonable range
    factor = max(1.0, min(2.0, factor))
    return enhancer.enhance(factor)


def enhance_image(image: Image.Image, 
                 auto_brightness: bool = True,
                 denoise: bool = True,
                 sharpen: bool = True,
                 brightness_threshold: float = 0.3,
                 target_brightness: float = 0.5) -> Tuple[Image.Image, dict]:
    """
    Apply comprehensive image enhancement.
    
    Args:
        image: PIL Image
        auto_brightness: Whether to auto-adjust brightness/contrast
        denoise: Whether to apply denoising
        sharpen: Whether to apply sharpening
        brightness_threshold: Minimum brightness threshold
        target_brightness: Target brightness level
        
    Returns:
        Tuple of (enhanced_image, enhancement_info)
    """
    # Create a copy to avoid modifying original
    enhanced = image.copy()
    info = {
        'original_brightness': calculate_brightness(image),
        'brightness_adjusted': False,
        'denoised': False,
        'sharpened': False
    }
    
    # Step 1: Auto brightness/contrast adjustment
    if auto_brightness:
        original_brightness = info['original_brightness']
        if original_brightness < brightness_threshold:
            enhanced = auto_adjust_brightness_contrast(
                enhanced, 
                target_brightness=target_brightness,
                brightness_threshold=brightness_threshold
            )
            info['brightness_adjusted'] = True
            info['new_brightness'] = calculate_brightness(enhanced)
    
    # Step 2: Denoising
    if denoise:
        enhanced = denoise_image(enhanced, method='median')
        info['denoised'] = True
    
    # Step 3: Sharpening (applied after denoising to restore edge details)
    if sharpen:
        enhanced = sharpen_image(enhanced, factor=1.2)
        info['sharpened'] = True
    
    return enhanced, info


def compare_images(original: Image.Image, enhanced: Image.Image) -> Image.Image:
    """
    Create a side-by-side comparison of original and enhanced images.
    
    Args:
        original: Original PIL Image
        enhanced: Enhanced PIL Image
        
    Returns:
        Side-by-side comparison image
    """
    # Ensure both images are the same size
    width, height = original.size
    enhanced = enhanced.resize((width, height), Image.Resampling.LANCZOS)
    
    # Create comparison image
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(original, (0, 0))
    comparison.paste(enhanced, (width, 0))
    
    return comparison

