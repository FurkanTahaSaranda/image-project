"""
Utility Functions Module
Helper functions for visualization, ROI extraction, and result display
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def extract_roi(image, x, y, width, height):
    """
    Extract Region of Interest (ROI) from an image.
    
    Args:
        image: Input image
        x, y: Top-left corner coordinates
        width, height: ROI dimensions
        
    Returns:
        roi: Extracted region of interest
    """
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x = max(0, min(x, w))
    y = max(0, min(y, h))
    width = min(width, w - x)
    height = min(height, h - y)
    
    roi = image[y:y+height, x:x+width]
    return roi


def draw_thermal_regions(image, regions, color=(0, 0, 255), thickness=2):
    """
    Draw bounding boxes around detected thermal regions.
    
    Args:
        image: Input image (will be modified)
        regions: List of detected regions with 'bbox' key
        color: BGR color for bounding boxes
        thickness: Line thickness
        
    Returns:
        image: Image with drawn bounding boxes
    """
    result_image = image.copy()
    
    for region in regions:
        x, y, w, h = region['bbox']
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
        
        # Add area label
        area = region['area']
        label = f"Area: {area}"
        cv2.putText(result_image, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return result_image


def overlay_thermal_mask(image, mask, alpha=0.5, color=(0, 0, 255)):
    """
    Overlay thermal mask on original image.
    
    Args:
        image: Original image
        mask: Binary thermal mask
        alpha: Transparency factor (0-1)
        color: BGR color for thermal regions
        
    Returns:
        overlay: Image with thermal mask overlay
    """
    overlay = image.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend images
    result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    
    return result


def visualize_processing_steps(original, hsv, blurred, mask, result, save_path=None):
    """
    Create a comprehensive visualization showing all processing steps.
    
    Args:
        original: Original RGB image
        hsv: HSV converted image (converted back to RGB for display)
        blurred: Blurred image
        mask: Thermal mask
        result: Final result with overlays
        save_path: Path to save the visualization (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # HSV image (show Value channel as grayscale)
    axes[0, 1].imshow(hsv[:, :, 2], cmap='gray')
    axes[0, 1].set_title('HSV - Value Channel', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Blurred image
    if len(blurred.shape) == 3:
        axes[0, 2].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title('Gaussian Blurred', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Thermal mask
    axes[1, 0].imshow(mask, cmap='hot')
    axes[1, 0].set_title('Thermal Mask', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Result with overlay
    axes[1, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Detection Result', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Combined view
    axes[1, 2].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[1, 2].imshow(mask, alpha=0.3, cmap='hot')
    axes[1, 2].set_title('Overlay View', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_grid(images, titles, rows=2, cols=2, save_path=None):
    """
    Create a grid comparison of multiple images.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        rows: Number of rows in grid
        cols: Number of columns in grid
        save_path: Path to save the comparison (optional)
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i >= len(axes):
            break
        
        if len(img.shape) == 3:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(img, cmap='gray')
        
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def add_text_overlay(image, text, position=(10, 30), color=(0, 255, 0), 
                     font_scale=0.7, thickness=2):
    """
    Add text overlay to image.
    
    Args:
        image: Input image
        text: Text to add
        position: (x, y) position for text
        color: BGR color for text
        font_scale: Font scale
        thickness: Text thickness
        
    Returns:
        image: Image with text overlay
    """
    result = image.copy()
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness)
    return result


def save_frame(frame, path, frame_number=None):
    """
    Save a frame to disk.
    
    Args:
        frame: Frame to save
        path: Directory path to save frame
        frame_number: Optional frame number for filename
    """
    import os
    os.makedirs(path, exist_ok=True)
    
    if frame_number is not None:
        filename = f"frame_{frame_number:06d}.jpg"
    else:
        import time
        filename = f"frame_{int(time.time())}.jpg"
    
    filepath = os.path.join(path, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def calculate_statistics(image, mask=None):
    """
    Calculate image statistics.
    
    Args:
        image: Input image
        mask: Optional mask to calculate statistics only for masked region
        
    Returns:
        stats: Dictionary of statistics
    """
    if mask is not None:
        pixels = image[mask > 0]
    else:
        pixels = image.flatten()
    
    if len(pixels) == 0:
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'median': 0
        }
    
    stats = {
        'mean': np.mean(pixels),
        'std': np.std(pixels),
        'min': np.min(pixels),
        'max': np.max(pixels),
        'median': np.median(pixels)
    }
    
    return stats

