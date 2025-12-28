"""
Smoke Detection Module
Detects smoke using color analysis, motion detection, and edge characteristics
"""

import numpy as np
import cv2
from spatial_filters import gaussian_blur, sobel_edge_detection


def detect_smoke_color(hsv_image):
    """
    Detect smoke based on color characteristics.
    Smoke typically appears as gray-white regions with low saturation.
    
    Args:
        hsv_image: Input HSV image
        
    Returns:
        mask: Binary mask of potential smoke regions
    """
    H = hsv_image[:, :, 0]
    S = hsv_image[:, :, 1]
    V = hsv_image[:, :, 2]
    
    # Smoke/Buhar color characteristics:
    # - Low saturation (gray-white, not colorful)
    # - Medium to high value (brightness)
    # - Hue can vary but typically in white/gray range
    
    # Create mask for gray-white regions (low saturation, medium-high brightness)
    # Adjusted for better buhar (steam) detection
    smoke_mask = (
        (S <= 60) &  # Low saturation (gray-white) - slightly higher for buhar
        (V >= 80) &  # Medium to high brightness - lower threshold for buhar
        (V <= 250)   # Not too bright (to avoid white hot surfaces)
    )
    
    return smoke_mask.astype(np.uint8) * 255


def detect_smoke_motion(current_frame, previous_frame, threshold=30):
    """
    Detect smoke using frame difference (motion detection).
    Smoke moves and changes shape, creating differences between frames.
    
    Args:
        current_frame: Current grayscale frame
        previous_frame: Previous grayscale frame
        threshold: Difference threshold
        
    Returns:
        motion_mask: Binary mask of moving regions
    """
    if previous_frame is None:
        return np.zeros(current_frame.shape, dtype=np.uint8)
    
    # Convert to grayscale if needed
    if len(current_frame.shape) == 3:
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    else:
        current_gray = current_frame
    
    if len(previous_frame.shape) == 3:
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    else:
        previous_gray = previous_frame
    
    # Calculate frame difference
    diff = cv2.absdiff(current_gray, previous_gray)
    
    # Threshold to get moving regions
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply Gaussian blur to smooth the mask
    motion_mask = gaussian_blur(motion_mask, kernel_size=5, sigma=1.5)
    _, motion_mask = cv2.threshold(motion_mask, 50, 255, cv2.THRESH_BINARY)
    
    return motion_mask.astype(np.uint8)


def detect_smoke_edges(image):
    """
    Detect smoke using edge characteristics.
    Smoke has soft, blurred edges compared to solid objects.
    
    Args:
        image: Input grayscale image
        
    Returns:
        edge_mask: Binary mask highlighting soft edge regions
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply edge detection
    edges, _ = sobel_edge_detection(gray)
    
    # Smoke has soft edges, so we look for regions with:
    # - Medium edge strength (not too strong, not too weak)
    # - After blurring, edges should be present but not sharp
    
    # Blur the edges to find soft edge regions
    blurred_edges = gaussian_blur(edges, kernel_size=7, sigma=2.0)
    
    # Threshold for medium edge strength (smoke characteristics)
    _, edge_mask = cv2.threshold(blurred_edges, 30, 255, cv2.THRESH_BINARY)
    _, edge_mask_upper = cv2.threshold(blurred_edges, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Combine: medium edge strength (soft edges)
    soft_edge_mask = cv2.bitwise_and(edge_mask, edge_mask_upper)
    
    return soft_edge_mask.astype(np.uint8)


def detect_smoke_regions(hsv_image, current_frame, previous_frame=None, 
                        min_area=200, use_motion=True, use_edges=True):
    """
    Comprehensive smoke detection combining multiple techniques.
    
    Args:
        hsv_image: Input HSV image
        current_frame: Current BGR frame
        previous_frame: Previous BGR frame for motion detection (optional)
        min_area: Minimum area for smoke regions (in pixels)
        use_motion: Whether to use motion detection
        use_edges: Whether to use edge characteristics
        
    Returns:
        mask: Binary mask of detected smoke regions
        contours: List of contours for detected regions
        regions: List of bounding boxes for detected regions
    """
    # Method 1: Color-based detection
    color_mask = detect_smoke_color(hsv_image)
    
    # Method 2: Motion-based detection
    motion_mask = np.zeros(color_mask.shape, dtype=np.uint8)
    if use_motion and previous_frame is not None:
        motion_mask = detect_smoke_motion(current_frame, previous_frame)
    
    # Method 3: Edge-based detection
    edge_mask = np.zeros(color_mask.shape, dtype=np.uint8)
    if use_edges:
        edge_mask = detect_smoke_edges(current_frame)
    
    # Combine masks (smoke should satisfy multiple criteria)
    # Combine color and motion (smoke moves and is gray-white)
    combined_mask = cv2.bitwise_and(color_mask, motion_mask) if use_motion else color_mask
    
    # Also include edge information (soft edges are characteristic of smoke)
    if use_edges:
        # Smoke regions should have soft edges
        combined_mask = cv2.bitwise_or(combined_mask, 
                                      cv2.bitwise_and(color_mask, edge_mask))
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    # Opening to remove small noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # Closing to fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Dilation to expand regions slightly
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
    
    return combined_mask, filtered_contours, regions


def calculate_smoke_density(mask, region_mask=None):
    """
    Calculate smoke density in detected regions.
    
    Args:
        mask: Binary smoke mask
        region_mask: Optional mask for specific region
        
    Returns:
        density: Smoke density (0-1)
        coverage: Percentage of frame covered by smoke
    """
    if region_mask is not None:
        smoke_pixels = np.sum(cv2.bitwise_and(mask, region_mask) > 0)
        total_pixels = np.sum(region_mask > 0)
    else:
        smoke_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
    
    if total_pixels == 0:
        return 0.0, 0.0
    
    density = smoke_pixels / total_pixels
    coverage = (smoke_pixels / total_pixels) * 100
    
    return density, coverage


def classify_smoke_severity(density, coverage, num_regions):
    """
    Classify smoke severity based on density and coverage.
    
    Args:
        density: Smoke density (0-1)
        coverage: Percentage of frame covered by smoke
        num_regions: Number of detected smoke regions
        
    Returns:
        severity: Severity level ('low', 'medium', 'high', 'critical')
        risk_score: Risk score from 0 to 100
    """
    # Calculate risk score
    density_score = min(100, density * 200)  # Density contributes up to 100
    coverage_score = min(100, coverage * 2)   # Coverage contributes up to 100
    region_score = min(50, num_regions * 10)  # Number of regions contributes up to 50
    
    # Combined risk score (weighted)
    risk_score = min(100, density_score * 0.4 + coverage_score * 0.5 + region_score * 0.1)
    
    # Classify severity
    if risk_score >= 70:
        severity = 'critical'
    elif risk_score >= 50:
        severity = 'high'
    elif risk_score >= 30:
        severity = 'medium'
    else:
        severity = 'low'
    
    return severity, risk_score

