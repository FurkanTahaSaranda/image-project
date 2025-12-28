"""
Main Thermal Anomaly Detection System
Real-time video processing pipeline for kitchen thermal detection
"""

import cv2
import numpy as np
import argparse
import os
import time
from datetime import datetime

# Import custom modules
from color_space import rgb_to_hsv, extract_thermal_hsv_ranges
from spatial_filters import gaussian_blur, sobel_edge_detection
from histogram_analysis import adaptive_threshold_thermal, plot_hsv_histograms
from threshold_detection import detect_thermal_regions, calculate_thermal_intensity, classify_thermal_severity
from smoke_detection import (detect_smoke_regions, calculate_smoke_density, classify_smoke_severity)
from utils import (draw_thermal_regions, overlay_thermal_mask, visualize_processing_steps,
                   add_text_overlay, save_frame, calculate_statistics)


class ThermalDetectionSystem:
    """Main thermal anomaly detection system class."""
    
    def __init__(self, use_adaptive=False, use_spatial_filter=True, min_area=100, 
                 enable_smoke_detection=True, smoke_min_area=150, process_every_n_frames=2,
                 resize_factor=0.6, use_fast_mode=True):
        """
        Initialize thermal detection system.
        
        Args:
            use_adaptive: Use adaptive thresholding based on histogram analysis
            use_spatial_filter: Apply Gaussian blur for noise reduction
            min_area: Minimum area for thermal regions (in pixels)
            enable_smoke_detection: Enable smoke detection
            smoke_min_area: Minimum area for smoke regions (in pixels)
            process_every_n_frames: Process every Nth frame (1=all frames, 2=every other frame, etc.)
            resize_factor: Resize factor for processing (1.0=original, 0.5=half size for speed)
            use_fast_mode: Use OpenCV optimized functions for better performance
        """
        self.use_adaptive = use_adaptive
        self.use_spatial_filter = use_spatial_filter
        self.min_area = min_area
        self.enable_smoke_detection = enable_smoke_detection
        self.smoke_min_area = smoke_min_area
        self.process_every_n_frames = process_every_n_frames
        self.resize_factor = resize_factor
        self.use_fast_mode = use_fast_mode
        
        # Get thermal HSV ranges
        self.thermal_ranges = extract_thermal_hsv_ranges()
        
        # Statistics tracking
        self.frame_count = 0
        self.detection_history = []
        
        # Previous frame for smoke motion detection
        self.previous_frame = None
        
        # Cache for last processed results (for frame skipping)
        self.last_result = None
        
    def process_frame(self, frame):
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input BGR frame from camera/video
            
        Returns:
            result: Processed frame with thermal regions highlighted
            mask: Binary thermal mask
            regions: Detected thermal regions
            severity: Thermal severity level
            risk_score: Risk score (0-100)
        """
        self.frame_count += 1
        
        # Frame skipping for performance
        if self.frame_count % self.process_every_n_frames != 0:
            if self.last_result is not None:
                # Return cached result
                return self.last_result
        
        original_frame = frame.copy()
        
        # Resize for faster processing if needed
        if self.resize_factor != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * self.resize_factor)
            new_height = int(height * self.resize_factor)
            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame
            new_width, new_height = frame.shape[1], frame.shape[0]
        
        # Step 1: Apply spatial filtering (Gaussian blur) for noise reduction
        # Apply to all frames but with smaller kernel for performance
        if self.use_spatial_filter:
            if self.use_fast_mode:
                # Use OpenCV's optimized Gaussian blur for better performance
                blurred_frame = cv2.GaussianBlur(frame_resized, (3, 3), 0.6)
            else:
                blurred_frame = gaussian_blur(frame_resized, kernel_size=3, sigma=0.6)
        else:
            blurred_frame = frame_resized.copy()
        
        # Step 2: Convert RGB to HSV color space
        if self.use_fast_mode:
            # Use OpenCV's optimized conversion for better performance
            hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        else:
            # Use manual implementation (from scratch)
            hsv_frame = rgb_to_hsv(blurred_frame)
        
        # Step 3: Adaptive thresholding based on histogram analysis
        # Skip adaptive thresholding most frames for performance (calculate every 5 frames)
        adaptive_thresholds = None
        if self.use_adaptive and (self.frame_count % 5 == 0 or not hasattr(self, '_last_adaptive_thresholds')):
            adaptive_thresholds = adaptive_threshold_thermal(hsv_frame)
            self._last_adaptive_thresholds = adaptive_thresholds
        elif self.use_adaptive:
            adaptive_thresholds = getattr(self, '_last_adaptive_thresholds', None)
        
        # Step 4: Detect thermal regions using threshold-based detection
        # Apply morphological operations to all frames but with reduced iterations
        mask, contours, regions = detect_thermal_regions(
            hsv_frame,
            thermal_ranges=self.thermal_ranges if not self.use_adaptive else None,
            adaptive_thresholds=adaptive_thresholds,
            min_area=self.min_area,
            apply_morphology=True  # Apply to all frames for better detection
        )
        
        # Step 5: Calculate thermal intensity
        intensity_map, avg_intensity = calculate_thermal_intensity(hsv_frame, mask)
        
        # Step 6: Classify thermal severity
        severity, risk_score = classify_thermal_severity(avg_intensity, regions)
        
        # Step 7: Detect smoke (if enabled)
        smoke_mask = None
        smoke_regions = []
        smoke_severity = 'low'
        smoke_risk_score = 0.0
        
        if self.enable_smoke_detection:
            # Resize previous frame if needed
            prev_frame = None
            if self.previous_frame is not None:
                if self.resize_factor != 1.0:
                    prev_frame = cv2.resize(self.previous_frame, (new_width, new_height), 
                                          interpolation=cv2.INTER_LINEAR)
                else:
                    prev_frame = self.previous_frame
            
            # Use motion detection more frequently for better smoke/buhar detection
            # Edge detection can be less frequent for performance
            use_edges = (self.frame_count % 2 == 0)  # Every other frame
            use_motion = True  # Always use motion for smoke/buhar detection
            smoke_mask, smoke_contours, smoke_regions = detect_smoke_regions(
                hsv_frame,
                blurred_frame,
                previous_frame=prev_frame,
                min_area=self.smoke_min_area,
                use_motion=use_motion,
                use_edges=use_edges
            )
            
            if len(smoke_regions) > 0:
                smoke_density, smoke_coverage = calculate_smoke_density(smoke_mask)
                smoke_severity, smoke_risk_score = classify_smoke_severity(
                    smoke_density, smoke_coverage, len(smoke_regions)
                )
        
        # Step 8: Visualize results
        # Resize masks back to original size if needed
        if self.resize_factor != 1.0:
            mask = cv2.resize(mask, (original_frame.shape[1], original_frame.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
            if smoke_mask is not None:
                smoke_mask = cv2.resize(smoke_mask, (original_frame.shape[1], original_frame.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            # Scale regions back to original size
            for region in regions:
                x, y, w, h = region['bbox']
                region['bbox'] = (int(x / self.resize_factor), int(y / self.resize_factor), 
                                 int(w / self.resize_factor), int(h / self.resize_factor))
            for region in smoke_regions:
                x, y, w, h = region['bbox']
                region['bbox'] = (int(x / self.resize_factor), int(y / self.resize_factor), 
                                 int(w / self.resize_factor), int(h / self.resize_factor))
        
        result_frame = overlay_thermal_mask(original_frame, mask, alpha=0.4, color=(0, 0, 255))
        result_frame = draw_thermal_regions(result_frame, regions, color=(0, 255, 255), thickness=2)
        
        # Overlay smoke detection (if detected)
        if self.enable_smoke_detection and smoke_mask is not None and np.sum(smoke_mask) > 0:
            result_frame = overlay_thermal_mask(result_frame, smoke_mask, alpha=0.3, color=(128, 128, 128))
            result_frame = draw_thermal_regions(result_frame, smoke_regions, color=(200, 200, 200), thickness=2)
        
        # Add information overlay
        info_text = f"Thermal: {severity.upper()} ({risk_score:.1f}%) | Regions: {len(regions)}"
        result_frame = add_text_overlay(result_frame, info_text, position=(10, 30),
                                        color=(0, 255, 0), font_scale=0.7, thickness=2)
        
        # Add smoke information (if detected)
        if self.enable_smoke_detection and smoke_risk_score > 0:
            smoke_text = f"Smoke: {smoke_severity.upper()} ({smoke_risk_score:.1f}%) | Regions: {len(smoke_regions)}"
            result_frame = add_text_overlay(result_frame, smoke_text, position=(10, 60),
                                           color=(200, 200, 200), font_scale=0.7, thickness=2)
        
        # Add frame number
        frame_text = f"Frame: {self.frame_count}"
        y_pos = 90 if (self.enable_smoke_detection and smoke_risk_score > 0) else 60
        result_frame = add_text_overlay(result_frame, frame_text, position=(10, y_pos),
                                       color=(255, 255, 255), font_scale=0.6, thickness=1)
        
        # Store detection history
        history_entry = {
            'frame': self.frame_count,
            'severity': severity,
            'risk_score': risk_score,
            'num_regions': len(regions),
            'avg_intensity': avg_intensity
        }
        
        if self.enable_smoke_detection:
            history_entry.update({
                'smoke_severity': smoke_severity,
                'smoke_risk_score': smoke_risk_score,
                'smoke_regions': len(smoke_regions)
            })
        
        self.detection_history.append(history_entry)
        
        # Update previous frame for motion detection (use resized version)
        self.previous_frame = blurred_frame.copy()
        
        # Cache result for frame skipping
        result = (result_frame, mask, regions, severity, risk_score, smoke_mask, smoke_regions)
        self.last_result = result
        
        return result
    
    def process_video(self, video_path, output_path=None, save_frames=False, 
                     display=True, max_frames=None):
        """
        Process video file frame by frame.
        
        Args:
            video_path: Path to input video file or camera index (0 for webcam)
            output_path: Path to save output video (optional)
            save_frames: Whether to save individual processed frames
            display: Whether to display frames in real-time
            max_frames: Maximum number of frames to process (None for all)
        """
        # Open video source
        if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
            cap = cv2.VideoCapture(int(video_path))
            is_camera = True
        else:
            cap = cv2.VideoCapture(video_path)
            is_camera = False
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if not is_camera else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        print("Starting video processing...")
        print("Press 'q' to quit, 's' to save current frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_idx >= max_frames:
                    break
                
                # Process frame
                result_frame, mask, regions, severity, risk_score, smoke_mask, smoke_regions = self.process_frame(frame)
                
                # Save frame if requested
                if save_frames and frame_idx % 10 == 0:  # Save every 10th frame
                    save_frame(result_frame, 'results/processed_frames', frame_idx)
                
                # Write to output video
                if writer:
                    writer.write(result_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Thermal Detection', result_frame)
                    
                    # Use waitKey(1) for real-time, but don't block too long
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f'results/processed_frames/snapshot_{timestamp}.jpg', result_frame)
                        print(f"Frame saved: snapshot_{timestamp}.jpg")
                
                frame_idx += 1
                
                # Print progress
                if frame_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                    print(f"Processed {frame_idx} frames | FPS: {fps_actual:.2f} | "
                          f"Current Risk: {risk_score:.1f}% ({severity})")
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            print(f"\nProcessing complete!")
            print(f"Total frames: {frame_idx}")
            print(f"Total time: {elapsed:.2f} seconds")
            print(f"Average FPS: {frame_idx / elapsed if elapsed > 0 else 0:.2f}")
            
            # Print detection statistics
            if self.detection_history:
                avg_risk = np.mean([d['risk_score'] for d in self.detection_history])
                max_risk = np.max([d['risk_score'] for d in self.detection_history])
                critical_frames = sum(1 for d in self.detection_history if d['severity'] == 'critical')
                
                print(f"\nThermal Detection Statistics:")
                print(f"  Average Risk Score: {avg_risk:.2f}%")
                print(f"  Maximum Risk Score: {max_risk:.2f}%")
                print(f"  Critical Frames: {critical_frames} ({100*critical_frames/len(self.detection_history):.1f}%)")
                
                # Smoke detection statistics
                if self.enable_smoke_detection and 'smoke_risk_score' in self.detection_history[0]:
                    smoke_frames = sum(1 for d in self.detection_history if d.get('smoke_risk_score', 0) > 0)
                    if smoke_frames > 0:
                        avg_smoke_risk = np.mean([d.get('smoke_risk_score', 0) for d in self.detection_history])
                        max_smoke_risk = np.max([d.get('smoke_risk_score', 0) for d in self.detection_history])
                        critical_smoke = sum(1 for d in self.detection_history if d.get('smoke_severity') == 'critical')
                        
                        print(f"\nSmoke Detection Statistics:")
                        print(f"  Frames with Smoke: {smoke_frames} ({100*smoke_frames/len(self.detection_history):.1f}%)")
                        print(f"  Average Smoke Risk: {avg_smoke_risk:.2f}%")
                        print(f"  Maximum Smoke Risk: {max_smoke_risk:.2f}%")
                        print(f"  Critical Smoke Frames: {critical_smoke} ({100*critical_smoke/len(self.detection_history):.1f}%)")
    
    def process_image(self, image_path, output_path=None, display=True):
        """
        Process a single image file.
        
        Args:
            image_path: Path to input image file
            output_path: Path to save output image (optional)
            display: Whether to display the result
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Process the image
        result_frame, mask, regions, severity, risk_score, smoke_mask, smoke_regions = self.process_frame(image)
        
        # Calculate smoke statistics if smoke detected
        smoke_severity = 'low'
        smoke_risk_score = 0.0
        if self.enable_smoke_detection and smoke_mask is not None and np.sum(smoke_mask) > 0:
            smoke_density, smoke_coverage = calculate_smoke_density(smoke_mask)
            smoke_severity, smoke_risk_score = classify_smoke_severity(
                smoke_density, smoke_coverage, len(smoke_regions)
            )
        
        # Print results
        print(f"\nDetection Results:")
        print(f"  Thermal Severity: {severity.upper()}")
        print(f"  Thermal Risk Score: {risk_score:.1f}%")
        print(f"  Thermal Regions Found: {len(regions)}")
        
        if self.enable_smoke_detection:
            if len(smoke_regions) > 0:
                print(f"  Smoke Severity: {smoke_severity.upper()}")
                print(f"  Smoke Risk Score: {smoke_risk_score:.1f}%")
                print(f"  Smoke Regions Found: {len(smoke_regions)}")
                if smoke_mask is not None:
                    smoke_density, smoke_coverage = calculate_smoke_density(smoke_mask)
                    print(f"  Smoke Coverage: {smoke_coverage:.1f}%")
            else:
                print(f"  Smoke: No smoke detected")
        
        # Save output image
        if output_path:
            cv2.imwrite(output_path, result_frame)
            print(f"\nProcessed image saved to: {output_path}")
        else:
            # Auto-generate output path
            import os
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f'results/processed_frames/{base_name}_processed.jpg'
            os.makedirs('results/processed_frames', exist_ok=True)
            cv2.imwrite(output_path, result_frame)
            print(f"\nProcessed image saved to: {output_path}")
        
        # Display result
        if display:
            cv2.imshow('Thermal Detection - Image', result_frame)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """Main function to run thermal detection system."""
    parser = argparse.ArgumentParser(description='Thermal Anomaly Detection in Kitchen Environments')
    parser.add_argument('--input', type=str, default='0',
                       help='Input video path, image path, or camera index (default: 0 for webcam)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--enable-adaptive', action='store_true',
                       help='Enable adaptive thresholding (disabled by default for performance)')
    parser.add_argument('--no-filter', action='store_true',
                       help='Disable spatial filtering')
    parser.add_argument('--min-area', type=int, default=100,
                       help='Minimum area for thermal regions (default: 100)')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save processed frames to disk')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--enable-smoke', action='store_true',
                       help='Enable smoke detection (disabled by default for performance)')
    parser.add_argument('--smoke-min-area', type=int, default=200,
                       help='Minimum area for smoke regions (default: 200)')
    parser.add_argument('--frame-skip', type=int, default=2,
                       help='Process every Nth frame (default: 2 for balanced speed/quality, 1=all frames)')
    parser.add_argument('--resize', type=float, default=0.6,
                       help='Resize factor for processing (default: 0.6 for balanced speed/quality, 1.0=original)')
    parser.add_argument('--no-fast', action='store_true',
                       help='Disable fast mode (use manual implementations, slower)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('results/processed_frames', exist_ok=True)
    os.makedirs('results/output_videos', exist_ok=True)
    
    # Initialize detection system
    system = ThermalDetectionSystem(
        use_adaptive=args.enable_adaptive,
        use_spatial_filter=not args.no_filter,
        min_area=args.min_area,
        enable_smoke_detection=args.enable_smoke,
        smoke_min_area=args.smoke_min_area,
        process_every_n_frames=args.frame_skip,
        resize_factor=args.resize,
        use_fast_mode=not args.no_fast
    )
    
    # Check if input is an image file
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    input_lower = args.input.lower()
    is_image = any(input_lower.endswith(ext) for ext in image_extensions)
    
    if is_image:
        # Process image
        output_image = args.output if args.output else None
        system.process_image(
            image_path=args.input,
            output_path=output_image,
            display=not args.no_display
        )
    else:
        # Process video or webcam
        # Set output path if not specified
        if args.output is None and args.input != '0':
            input_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f'results/output_videos/{input_name}_processed.mp4'
        
        # Process video
        system.process_video(
            video_path=args.input,
            output_path=args.output,
            save_frames=args.save_frames,
            display=not args.no_display,
            max_frames=args.max_frames
        )


if __name__ == '__main__':
    main()

