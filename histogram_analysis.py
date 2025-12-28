"""
Histogram Analysis Module
Histogram calculation and adaptive thresholding for thermal detection
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_histogram(image, bins=256):
    """
    Calculate histogram of an image.
    
    Args:
        image: Input image as numpy array (grayscale or single channel)
        bins: Number of bins for histogram
        
    Returns:
        hist: Histogram array
        bin_edges: Bin edges array
    """
    # Flatten image to 1D array
    flat_image = image.flatten()
    
    # Calculate histogram
    hist, bin_edges = np.histogram(flat_image, bins=bins, range=(0, 256))
    
    return hist, bin_edges


def calculate_cumulative_histogram(hist):
    """
    Calculate cumulative histogram.
    
    Args:
        hist: Histogram array
        
    Returns:
        cum_hist: Cumulative histogram
    """
    cum_hist = np.cumsum(hist)
    return cum_hist


def otsu_threshold(hist):
    """
    Calculate optimal threshold using Otsu's method.
    
    Args:
        hist: Histogram array
        
    Returns:
        threshold: Optimal threshold value
    """
    # Total number of pixels
    total_pixels = np.sum(hist)
    
    if total_pixels == 0:
        return 128  # Default threshold
    
    # Normalize histogram to get probabilities
    prob = hist / total_pixels
    
    # Initialize variables
    best_threshold = 0
    max_variance = 0
    
    # Try all possible thresholds
    for t in range(256):
        # Background class (0 to t)
        w0 = np.sum(prob[:t+1])
        if w0 == 0:
            continue
        
        # Foreground class (t+1 to 255)
        w1 = np.sum(prob[t+1:])
        if w1 == 0:
            continue
        
        # Mean of background class
        mean0 = np.sum(np.arange(t+1) * prob[:t+1]) / w0
        
        # Mean of foreground class
        mean1 = np.sum(np.arange(t+1, 256) * prob[t+1:]) / w1
        
        # Between-class variance
        variance = w0 * w1 * (mean0 - mean1) ** 2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    
    return best_threshold


def adaptive_threshold_hsv(hsv_image, channel='v', method='otsu'):
    """
    Calculate adaptive threshold for HSV image based on histogram analysis.
    
    Args:
        hsv_image: Input HSV image
        channel: Channel to analyze ('h', 's', or 'v')
        method: Threshold method ('otsu' or 'mean')
        
    Returns:
        threshold: Adaptive threshold value
    """
    # Extract channel
    if channel == 'h':
        channel_data = hsv_image[:, :, 0]
    elif channel == 's':
        channel_data = hsv_image[:, :, 1]
    else:  # 'v'
        channel_data = hsv_image[:, :, 2]
    
    # Calculate histogram
    hist, _ = calculate_histogram(channel_data)
    
    if method == 'otsu':
        threshold = otsu_threshold(hist)
    else:  # 'mean'
        threshold = np.mean(channel_data)
    
    return threshold


def adaptive_threshold_thermal(hsv_image):
    """
    Calculate adaptive thresholds for thermal detection using histogram analysis.
    Adapts to varying lighting conditions.
    
    Args:
        hsv_image: Input HSV image
        
    Returns:
        thresholds: Dictionary containing adaptive thresholds for each HSV channel
    """
    # Calculate histograms for each channel
    h_hist, _ = calculate_histogram(hsv_image[:, :, 0])
    s_hist, _ = calculate_histogram(hsv_image[:, :, 1])
    v_hist, _ = calculate_histogram(hsv_image[:, :, 2])
    
    # Calculate adaptive thresholds
    # For Value channel, use Otsu to separate bright (thermal) regions
    v_threshold = otsu_threshold(v_hist)
    
    # For Saturation, use mean to identify colorful regions
    s_threshold = np.mean(hsv_image[:, :, 1])
    
    # For Hue, we use fixed ranges but can adapt based on histogram peaks
    h_hist_peaks = np.where(h_hist > np.max(h_hist) * 0.1)[0]
    if len(h_hist_peaks) > 0:
        h_min = np.min(h_hist_peaks)
        h_max = np.max(h_hist_peaks)
    else:
        h_min = 0
        h_max = 179
    
    thresholds = {
        'h_min': max(0, h_min - 10),
        'h_max': min(179, h_max + 10),
        's_min': max(50, int(s_threshold * 0.5)),
        's_max': 255,
        'v_min': max(100, int(v_threshold * 0.7)),
        'v_max': 255
    }
    
    return thresholds


def plot_histogram(image, title="Histogram", save_path=None):
    """
    Plot and optionally save histogram of an image.
    
    Args:
        image: Input image
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    hist, bin_edges = calculate_histogram(image)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, hist, width=1, alpha=0.7)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_hsv_histograms(hsv_image, save_path=None):
    """
    Plot histograms for all HSV channels.
    
    Args:
        hsv_image: Input HSV image
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    channels = ['Hue', 'Saturation', 'Value']
    colors = ['red', 'green', 'blue']
    
    for i, (channel, color) in enumerate(zip(channels, colors)):
        hist, bin_edges = calculate_histogram(hsv_image[:, :, i])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        axes[i].bar(bin_centers, hist, width=1, alpha=0.7, color=color)
        axes[i].set_xlabel('Pixel Intensity')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{channel} Histogram')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

