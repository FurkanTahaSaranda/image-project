"""
Threshold-Based Thermal Detection Module
Detects thermal regions using HSV color space and thresholding
"""

import numpy as np
import cv2


def create_thermal_mask(hsv_image, thermal_ranges=None, adaptive_thresholds=None):
    """
    Create a binary mask for thermal regions based on HSV thresholds.
    
    Args:
        hsv_image: Input HSV image
        thermal_ranges: Dictionary of predefined HSV ranges (optional)
        adaptive_thresholds: Dictionary of adaptive thresholds (optional)
        
    Returns:
        mask: Binary mask where 1 indicates thermal regions
    """
    H = hsv_image[:, :, 0]
    S = hsv_image[:, :, 1]
    V = hsv_image[:, :, 2]
    
    # Initialize mask
    mask = np.zeros((hsv_image.shape[0], hsv_image.shape[1]), dtype=np.uint8)
    
    if adaptive_thresholds:
        # Use adaptive thresholds
        h_min = adaptive_thresholds['h_min']
        h_max = adaptive_thresholds['h_max']
        s_min = adaptive_thresholds['s_min']
        s_max = adaptive_thresholds['s_max']
        v_min = adaptive_thresholds['v_min']
        v_max = adaptive_thresholds['v_max']
        
        # Create mask for main range
        mask = ((H >= h_min) & (H <= h_max) & 
                (S >= s_min) & (S <= s_max) & 
                (V >= v_min) & (V <= v_max)).astype(np.uint8) * 255
        
        # Also check for red range (wraps around 0)
        if h_min > h_max:  # Wraps around
            red_mask = ((H >= h_min) | (H <= h_max)) & \
                       (S >= s_min) & (S <= s_max) & \
                       (V >= v_min) & (V <= v_max)
            mask = np.maximum(mask, (red_mask.astype(np.uint8) * 255))
    
    elif thermal_ranges:
        # Use predefined ranges
        # Red low range (0-10)
        red_low = thermal_ranges.get('red_low', {})
        if red_low:
            mask_red_low = ((H >= red_low['h_min']) & (H <= red_low['h_max']) &
                           (S >= red_low['s_min']) & (S <= red_low['s_max']) &
                           (V >= red_low['v_min']) & (V <= red_low['v_max']))
            mask = np.maximum(mask, mask_red_low.astype(np.uint8) * 255)
        
        # Red high range (170-179)
        red_high = thermal_ranges.get('red_high', {})
        if red_high:
            mask_red_high = ((H >= red_high['h_min']) & (H <= red_high['h_max']) &
                            (S >= red_high['s_min']) & (S <= red_high['s_max']) &
                            (V >= red_high['v_min']) & (V <= red_high['v_max']))
            mask = np.maximum(mask, mask_red_high.astype(np.uint8) * 255)
        
        # Yellow range (medium temperature)
        orange = thermal_ranges.get('orange', {})
        if orange:
            mask_orange = ((H >= orange['h_min']) & (H <= orange['h_max']) &
                          (S >= orange['s_min']) & (S <= orange['s_max']) &
                          (V >= orange['v_min']) & (V <= orange['v_max']))
            mask = np.maximum(mask, mask_orange.astype(np.uint8) * 255)
        
        # White range (lowest temperature - first warning)
        white_yellow = thermal_ranges.get('white_yellow', {})
        if white_yellow:
            mask_white = ((H >= white_yellow['h_min']) & (H <= white_yellow['h_max']) &
                         (S >= white_yellow['s_min']) & (S <= white_yellow['s_max']) &
                         (V >= white_yellow['v_min']) & (V <= white_yellow['v_max']))
            mask = np.maximum(mask, mask_white.astype(np.uint8) * 255)
    else:
        # Default thresholds
        # Red range (highest temperature - critical)
        mask_red_orange = ((H <= 25) | (H >= 170)) & (S >= 50) & (V >= 100)
        # White (lowest temperature - first warning)
        mask_white = (H >= 20) & (H <= 30) & (S <= 50) & (V >= 200)
        
        mask = np.maximum(mask_red_orange.astype(np.uint8) * 255,
                         mask_white.astype(np.uint8) * 255)
    
    return mask


def detect_thermal_regions(hsv_image, thermal_ranges=None, adaptive_thresholds=None,
                          min_area=100, apply_morphology=True):
    """
    Detect thermal regions in HSV image using threshold-based detection.
    
    Args:
        hsv_image: Input HSV image
        thermal_ranges: Predefined HSV ranges (optional)
        adaptive_thresholds: Adaptive thresholds from histogram analysis (optional)
        min_area: Minimum area for a region to be considered (in pixels)
        apply_morphology: Whether to apply morphological operations for noise reduction
        
    Returns:
        mask: Binary mask of thermal regions
        contours: List of contours for detected regions
        regions: List of bounding boxes for detected regions
    """
    # Create thermal mask
    mask = create_thermal_mask(hsv_image, thermal_ranges, adaptive_thresholds)
    
    # Apply morphological operations to reduce noise
    if apply_morphology:
        # Use smaller kernel and fewer iterations for better performance
        kernel = np.ones((3, 3), np.uint8)
        # Simplified morphological operations (faster)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    filtered_contours = []
    regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            filtered_contours.append(contour)
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            regions.append({
                'bbox': (x, y, w, h),
                'area': area,
                'contour': contour
            })
    
    return mask, filtered_contours, regions


def calculate_thermal_intensity(hsv_image, mask):
    """
    Calculate thermal intensity for detected regions.
    
    Args:
        hsv_image: Input HSV image
        mask: Binary mask of thermal regions
        
    Returns:
        intensity_map: Map of thermal intensity values
        avg_intensity: Average thermal intensity
    """
    # Use Value channel as intensity indicator
    V = hsv_image[:, :, 2]
    
    # Calculate intensity only for thermal regions
    intensity_map = np.where(mask > 0, V.astype(np.float32), 0)
    
    # Calculate average intensity
    thermal_pixels = np.sum(mask > 0)
    if thermal_pixels > 0:
        avg_intensity = np.sum(intensity_map) / thermal_pixels
    else:
        avg_intensity = 0
    
    return intensity_map, avg_intensity


def classify_thermal_severity(avg_intensity, regions):
    """
    Classify thermal severity based on intensity and region characteristics.
    
    Args:
        avg_intensity: Average thermal intensity
        regions: List of detected thermal regions
        
    Returns:
        severity: Severity level ('low', 'medium', 'high', 'critical')
        risk_score: Risk score from 0 to 100
    """
    # Calculate risk score based on intensity and number/size of regions
    intensity_score = min(100, (avg_intensity / 255.0) * 100)
    
    # Calculate region score
    num_regions = len(regions)
    total_area = sum(r['area'] for r in regions)
    region_score = min(50, (num_regions * 10) + (total_area / 1000))
    
    # Combined risk score
    risk_score = min(100, intensity_score * 0.7 + region_score * 0.3)
    
    # Classify severity
    if risk_score >= 80:
        severity = 'critical'
    elif risk_score >= 60:
        severity = 'high'
    elif risk_score >= 40:
        severity = 'medium'
    else:
        severity = 'low'
    
    return severity, risk_score

