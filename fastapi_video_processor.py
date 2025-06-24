"""
FastAPI Video Processor Module

Handles video processing operations including frame extraction and feature computation
for the FastAPI video frame processing application.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Optional

class VideoProcessor:
    """Handles video processing operations including frame extraction and feature computation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_frames(self, video_path: str, output_dir: str, video_id: str, 
                      interval: float = 1.0) -> List[Dict]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to the input video file
            output_dir: Directory to save extracted frames
            video_id: Unique identifier for the video
            interval: Time interval between frames in seconds
            
        Returns:
            List of dictionaries containing frame information
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_path}")
                return []
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            self.logger.info(f"Video info - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
            
            # Calculate frame interval
            frame_interval = int(fps * interval) if fps > 0 else 1
            
            frames_info = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified intervals
                if frame_count % frame_interval == 0:
                    # Generate unique frame ID
                    frame_id = f"{video_id}_{extracted_count:04d}"
                    timestamp = frame_count / fps if fps > 0 else extracted_count * interval
                    
                    # Save frame as JPEG
                    frame_filename = f"{frame_id}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    # Resize frame if too large (optional optimization)
                    height, width = frame.shape[:2]
                    if width > 1920 or height > 1080:
                        scale = min(1920/width, 1080/height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    success = cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if success:
                        frames_info.append({
                            'id': frame_id,
                            'path': frame_path,
                            'timestamp': timestamp,
                            'frame_number': frame_count,
                            'video_id': video_id
                        })
                        extracted_count += 1
                        self.logger.debug(f"Extracted frame {extracted_count} at {timestamp:.2f}s")
                
                frame_count += 1
            
            cap.release()
            self.logger.info(f"Successfully extracted {len(frames_info)} frames from video")
            return frames_info
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def compute_feature_vector(self, frame_path: str) -> Optional[np.ndarray]:
        """
        Compute feature vector for a frame using color histograms and texture features.
        
        Args:
            frame_path: Path to the frame image
            
        Returns:
            Feature vector as numpy array or None if error
        """
        try:
            # Read image
            image = cv2.imread(frame_path)
            if image is None:
                self.logger.error(f"Could not read image: {frame_path}")
                return None
            
            # Convert BGR to RGB for better color representation
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Compute color histograms for each channel
            hist_r = cv2.calcHist([image_rgb], [0], None, [64], [0, 256])
            hist_g = cv2.calcHist([image_rgb], [1], None, [64], [0, 256])
            hist_b = cv2.calcHist([image_rgb], [2], None, [64], [0, 256])
            
            # Flatten and normalize histograms
            hist_r = hist_r.flatten() / (image.shape[0] * image.shape[1])
            hist_g = hist_g.flatten() / (image.shape[0] * image.shape[1])
            hist_b = hist_b.flatten() / (image.shape[0] * image.shape[1])
            
            # Combine histograms into single feature vector
            feature_vector = np.concatenate([hist_r, hist_g, hist_b])
            
            # Additional features: basic statistics
            mean_color = np.mean(image_rgb, axis=(0, 1))
            std_color = np.std(image_rgb, axis=(0, 1))
            
            # Convert to grayscale for additional features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Texture features using gradient
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            texture_features = np.array([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.mean(gray),
                np.std(gray)
            ])
            
            # Combine all features
            final_feature_vector = np.concatenate([
                feature_vector,  # Color histograms (192 features)
                mean_color,      # Mean RGB (3 features)
                std_color,       # Std RGB (3 features)
                texture_features # Texture features (4 features)
            ])
            
            # Normalize the feature vector
            norm = np.linalg.norm(final_feature_vector)
            if norm > 0:
                final_feature_vector = final_feature_vector / norm
            
            self.logger.debug(f"Computed feature vector of size {len(final_feature_vector)} for {frame_path}")
            return final_feature_vector.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error computing feature vector for {frame_path}: {str(e)}")
            return None
    
    def batch_compute_features(self, frame_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute feature vectors for multiple frames.
        
        Args:
            frame_paths: List of frame file paths
            
        Returns:
            Dictionary mapping frame paths to feature vectors
        """
        features = {}
        for frame_path in frame_paths:
            feature_vector = self.compute_feature_vector(frame_path)
            if feature_vector is not None:
                features[frame_path] = feature_vector
        
        self.logger.info(f"Computed features for {len(features)}/{len(frame_paths)} frames")
        return features
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get basic information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': duration
            }
            
        except Exception as e:
            self.logger.error(f"Error getting video info: {str(e)}")
            return {}