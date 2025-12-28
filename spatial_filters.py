"""
Spatial Filtering Module
Manual kernel convolution for Gaussian blur and edge detection
"""

import numpy as np


def create_gaussian_kernel(size, sigma):
    """
    Create a Gaussian kernel from scratch.
    
    Formula: G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))
    
    Args:
        size: Kernel size (must be odd, e.g., 3, 5, 7)
        sigma: Standard deviation of Gaussian distribution
        
    Returns:
        kernel: 2D Gaussian kernel normalized so sum equals 1
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size
    
    # Create coordinate arrays
    center = size // 2
    x = np.arange(size) - center
    y = np.arange(size) - center
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Calculate Gaussian values
    coefficient = 1 / (2 * np.pi * sigma ** 2)
    exponent = -(X ** 2 + Y ** 2) / (2 * sigma ** 2)
    kernel = coefficient * np.exp(exponent)
    
    # Normalize kernel so sum equals 1
    kernel = kernel / np.sum(kernel)
    
    return kernel


def apply_convolution(image, kernel, padding='zero'):
    """
    Apply manual convolution operation to an image.
    
    Args:
        image: Input image as numpy array (H, W) or (H, W, C)
        kernel: Convolution kernel as numpy array (K, K)
        padding: Padding mode - 'zero' for zero-padding, 'same' for same-size output
        
    Returns:
        convolved: Convolved image with same shape as input (if padding='same')
    """
    # Handle grayscale and color images
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]  # Add channel dimension
        is_grayscale = True
    else:
        is_grayscale = False
    
    kernel_height, kernel_width = kernel.shape
    image_height, image_width, num_channels = image.shape
    
    # Calculate padding
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    
    # Apply zero padding
    if padding == 'zero':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    else:
        padded = image
    
    # Initialize output
    output = np.zeros((image_height, image_width, num_channels), dtype=np.float32)
    
    # Flip kernel for convolution (correlation)
    kernel_flipped = np.flip(np.flip(kernel, 0), 1)
    
    # Apply convolution
    for c in range(num_channels):
        for i in range(image_height):
            for j in range(image_width):
                # Extract region of interest
                roi = padded[i:i + kernel_height, j:j + kernel_width, c]
                # Element-wise multiplication and sum
                output[i, j, c] = np.sum(roi * kernel_flipped)
    
    # Remove channel dimension if grayscale
    if is_grayscale:
        output = output[:, :, 0]
    
    # Clip values to valid range
    if image.dtype == np.uint8:
        output = np.clip(output, 0, 255).astype(np.uint8)
    else:
        output = output.astype(image.dtype)
    
    return output


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur to an image using manual convolution.
    
    Args:
        image: Input image as numpy array
        kernel_size: Size of Gaussian kernel (odd number)
        sigma: Standard deviation for Gaussian
        
    Returns:
        blurred: Blurred image
    """
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma)
    
    # Apply convolution
    blurred = apply_convolution(image, kernel, padding='zero')
    
    return blurred


def create_sobel_kernels():
    """
    Create Sobel kernels for edge detection.
    
    Returns:
        sobel_x: Horizontal Sobel kernel (detects vertical edges)
        sobel_y: Vertical Sobel kernel (detects horizontal edges)
    """
    # Sobel X kernel (detects vertical edges)
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    # Sobel Y kernel (detects horizontal edges)
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)
    
    return sobel_x, sobel_y


def sobel_edge_detection(image):
    """
    Apply Sobel edge detection using manual convolution.
    
    Args:
        image: Input grayscale image as numpy array (H, W)
        
    Returns:
        magnitude: Edge magnitude image
        direction: Edge direction image (in radians)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        # Convert BGR to grayscale manually
        gray = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
        gray = gray.astype(np.uint8)
    else:
        gray = image.copy()
    
    # Create Sobel kernels
    sobel_x, sobel_y = create_sobel_kernels()
    
    # Apply convolution
    Gx = apply_convolution(gray.astype(np.float32), sobel_x, padding='zero')
    Gy = apply_convolution(gray.astype(np.float32), sobel_y, padding='zero')
    
    # Calculate edge magnitude
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    
    # Calculate edge direction
    direction = np.arctan2(Gy, Gx)
    
    # Normalize magnitude to 0-255 range
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude, direction


def apply_median_filter(image, kernel_size=3):
    """
    Apply median filter for noise reduction (alternative to Gaussian).
    
    Args:
        image: Input image as numpy array
        kernel_size: Size of median filter kernel (odd number)
        
    Returns:
        filtered: Median filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    pad = kernel_size // 2
    
    # Handle grayscale and color images
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        is_grayscale = True
    else:
        is_grayscale = False
    
    height, width, channels = image.shape
    output = np.zeros_like(image)
    
    # Apply padding
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    
    # Apply median filter
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                roi = padded[i:i + kernel_size, j:j + kernel_size, c]
                output[i, j, c] = np.median(roi)
    
    if is_grayscale:
        output = output[:, :, 0]
    
    return output.astype(image.dtype)

