"""
Color Space Conversion Module
RGB to HSV conversion implemented from scratch
"""

import numpy as np


def rgb_to_hsv(rgb_image):
    """
    Convert RGB image to HSV color space from scratch.
    
    Formula:
    - H (Hue): 0-360 degrees (normalized to 0-179 for OpenCV compatibility)
    - S (Saturation): 0-1 (normalized to 0-255)
    - V (Value): 0-1 (normalized to 0-255)
    
    Args:
        rgb_image: Input RGB image as numpy array (H, W, 3) with values 0-255
        
    Returns:
        hsv_image: HSV image as numpy array (H, W, 3) with H:0-179, S:0-255, V:0-255
    """
    # Normalize RGB values to 0-1 range
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    
    # Extract R, G, B channels
    R = rgb_normalized[:, :, 2]  # OpenCV uses BGR, so R is at index 2
    G = rgb_normalized[:, :, 1]
    B = rgb_normalized[:, :, 0]
    
    # Get image dimensions
    height, width = R.shape
    
    # Initialize HSV arrays
    H = np.zeros((height, width), dtype=np.float32)
    S = np.zeros((height, width), dtype=np.float32)
    V = np.zeros((height, width), dtype=np.float32)
    
    # Calculate Value (V) - maximum of R, G, B
    V = np.maximum(np.maximum(R, G), B)
    
    # Calculate Saturation (S)
    # S = (V == 0) ? 0 : (V - min(R,G,B)) / V
    min_rgb = np.minimum(np.minimum(R, G), B)
    delta = V - min_rgb
    
    # Avoid division by zero
    S = np.where(V == 0, 0, delta / (V + 1e-10))
    
    # Calculate Hue (H) - Vectorized for better performance
    # H calculation depends on which channel is maximum
    # Create masks for each case
    delta_safe = delta + 1e-10  # Avoid division by zero
    
    # Red is max
    mask_r = (V == R) & (delta > 0)
    H_r = np.where(mask_r, 60 * (((G - B) / delta_safe) % 6), 0)
    
    # Green is max
    mask_g = (V == G) & (V != R) & (delta > 0)
    H_g = np.where(mask_g, 60 * (((B - R) / delta_safe) + 2), 0)
    
    # Blue is max
    mask_b = (V == B) & (V != R) & (V != G) & (delta > 0)
    H_b = np.where(mask_b, 60 * (((R - G) / delta_safe) + 4), 0)
    
    # Combine all cases
    H = H_r + H_g + H_b
    
    # Normalize to 0-360
    H = np.where(H < 0, H + 360, H)
    
    # Convert H to 0-179 range (OpenCV format)
    H = (H / 2).astype(np.uint8)
    
    # Convert S and V to 0-255 range
    S = (S * 255).astype(np.uint8)
    V = (V * 255).astype(np.uint8)
    
    # Stack channels to form HSV image
    hsv_image = np.stack([H, S, V], axis=2)
    
    return hsv_image


def hsv_to_rgb(hsv_image):
    """
    Convert HSV image back to RGB color space (for visualization purposes).
    
    Args:
        hsv_image: Input HSV image as numpy array (H, W, 3)
        
    Returns:
        rgb_image: RGB image as numpy array (H, W, 3) with values 0-255
    """
    # Extract H, S, V channels
    H = hsv_image[:, :, 0].astype(np.float32) * 2  # Convert back to 0-360
    S = hsv_image[:, :, 1].astype(np.float32) / 255.0
    V = hsv_image[:, :, 2].astype(np.float32) / 255.0
    
    height, width = H.shape
    
    # Initialize RGB arrays
    R = np.zeros((height, width), dtype=np.float32)
    G = np.zeros((height, width), dtype=np.float32)
    B = np.zeros((height, width), dtype=np.float32)
    
    # Calculate RGB values
    C = V * S  # Chroma
    X = C * (1 - abs((H / 60) % 2 - 1))
    m = V - C
    
    for i in range(height):
        for j in range(width):
            h = H[i, j]
            c = C[i, j]
            x = X[i, j]
            m_val = m[i, j]
            
            if 0 <= h < 60:
                R[i, j] = c
                G[i, j] = x
                B[i, j] = 0
            elif 60 <= h < 120:
                R[i, j] = x
                G[i, j] = c
                B[i, j] = 0
            elif 120 <= h < 180:
                R[i, j] = 0
                G[i, j] = c
                B[i, j] = x
            elif 180 <= h < 240:
                R[i, j] = 0
                G[i, j] = x
                B[i, j] = c
            elif 240 <= h < 300:
                R[i, j] = x
                G[i, j] = 0
                B[i, j] = c
            else:  # 300 <= h < 360
                R[i, j] = c
                G[i, j] = 0
                B[i, j] = x
    
    # Add m to each channel and convert to 0-255
    R = ((R + m) * 255).astype(np.uint8)
    G = ((G + m) * 255).astype(np.uint8)
    B = ((B + m) * 255).astype(np.uint8)
    
    # Stack channels (BGR format for OpenCV)
    rgb_image = np.stack([B, G, R], axis=2)
    
    return rgb_image


def extract_thermal_hsv_ranges():
    """
    Define HSV ranges for thermal regions (red, yellow, white).
    Temperature order: White (lowest) -> Yellow (medium) -> Red (highest/critical).
    
    Returns:
        dict: Dictionary containing HSV ranges for different thermal colors
    """
    thermal_ranges = {
        # Red range (0-10 and 170-180 in hue) - HIGHEST TEMPERATURE (Critical)
        'red_low': {
            'h_min': 0,
            'h_max': 10,
            's_min': 50,
            's_max': 255,
            'v_min': 50,
            'v_max': 255
        },
        'red_high': {
            'h_min': 170,
            'h_max': 179,
            's_min': 50,
            's_max': 255,
            'v_min': 50,
            'v_max': 255
        },
        # Yellow range (10-25 in hue) - MEDIUM TEMPERATURE
        'orange': {
            'h_min': 10,
            'h_max': 25,
            's_min': 100,
            's_max': 255,
            'v_min': 100,
            'v_max': 255
        },
        # White range (high value, low saturation) - LOWEST TEMPERATURE (First Warning)
        'white_yellow': {
            'h_min': 20,
            'h_max': 30,
            's_min': 0,
            's_max': 50,
            'v_min': 200,
            'v_max': 255
        }
    }
    
    return thermal_ranges

