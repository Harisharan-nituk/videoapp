import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import os
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import imageio
import tempfile

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Comprehensive video processing utilities"""
    
    def __init__(self):
        """Initialize video processor"""
        self.temp_dir = tempfile.mkdtemp()
        self.supported_formats = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm']
        
        # Video processing parameters
        self.default_fps = 30
        self.default_quality = 'high'
        self.compression_settings = {
            'high': {'crf': 18, 'preset': 'slow'},
            'medium': {'crf': 23, 'preset': 'medium'},
            'low': {'crf': 28, 'preset': 'fast'}
        }
    
    def load_video(self, video_path: str) -> Dict:
        """
        Load video file and extract frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video data
        """
        try:
            logger.info(f"Loading video: {video_path}")
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Extract frames
            frames = []
            frame_indices = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                frame_indices.append(len(frames) - 1)
            
            cap.release()
            
            # Load audio information
            audio_info = self._extract_audio_info(video_path)
            
            video_data = {
                'frames': frames,
                'frame_indices': frame_indices,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'audio_info': audio_info,
                'original_path': video_path
            }
            
            logger.info(f"Loaded video: {frame_count} frames, {fps} FPS, {width}x{height}")
            return video_data
            
        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            raise
    
    def save_video(self, frames: List[np.ndarray], output_path: str, 
                   fps: float = None, quality: str = 'high', 
                   audio_path: str = None) -> None:
        """
        Save frames as video file
        
        Args:
            frames: List of video frames
            output_path: Output video path
            fps: Frames per second
            quality: Video quality ('high', 'medium', 'low')
            audio_path: Path to audio file to include
        """
        try:
            logger.info(f"Saving video: {output_path}")
            
            if not frames:
                raise ValueError("No frames to save")
            
            # Set default FPS if not provided
            if fps is None:
                fps = self.default_fps
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise ValueError(f"Could not open video writer for: {output_path}")
            
            # Write frames
            for i, frame in enumerate(frames):
                if i % 100 == 0:
                    logger.info(f"Processing frame {i}/{len(frames)}")
                
                # Ensure frame is in correct format
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    out.write(frame)
                else:
                    # Convert to BGR if needed
                    if len(frame.shape) == 2:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame_bgr = frame
                    out.write(frame_bgr)
            
            out.release()
            
            # Add audio if provided
            if audio_path and os.path.exists(audio_path):
                self._add_audio_to_video(output_path, audio_path)
            
            # Apply compression if needed
            if quality != 'high':
                self._compress_video(output_path, quality)
            
            logger.info(f"Video saved successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            raise
    
    def _extract_audio_info(self, video_path: str) -> Dict:
        """Extract audio information from video"""
        try:
            with VideoFileClip(video_path) as clip:
                if clip.audio is not None:
                    return {
                        'has_audio': True,
                        'duration': clip.audio.duration,
                        'fps': clip.audio.fps,
                        'channels': clip.audio.nchannels if hasattr(clip.audio, 'nchannels') else 2
                    }
                else:
                    return {'has_audio': False}
        except Exception as e:
            logger.warning(f"Could not extract audio info: {str(e)}")
            return {'has_audio': False}
    
    def _add_audio_to_video(self, video_path: str, audio_path: str) -> None:
        """Add audio track to video"""
        try:
            # Create temporary output path
            temp_output = video_path.replace('.mp4', '_with_audio.mp4')
            
            # Load video and audio
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Set audio to video
            final_clip = video_clip.set_audio(audio_clip)
            
            # Write final video
            final_clip.write_videofile(temp_output, verbose=False, logger=None)
            
            # Replace original file
            os.replace(temp_output, video_path)
            
            # Clean up
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
        except Exception as e:
            logger.error(f"Error adding audio: {str(e)}")
    
    def _compress_video(self, video_path: str, quality: str) -> None:
        """Compress video based on quality setting"""
        try:
            settings = self.compression_settings.get(quality, self.compression_settings['medium'])
            
            # Create temporary compressed file
            temp_output = video_path.replace('.mp4', '_compressed.mp4')
            
            # Use moviepy for compression
            with VideoFileClip(video_path) as clip:
                clip.write_videofile(
                    temp_output,
                    bitrate=f"{2000 if quality == 'high' else 1000 if quality == 'medium' else 500}k",
                    verbose=False,
                    logger=None
                )
            
            # Replace original file
            os.replace(temp_output, video_path)
            
        except Exception as e:
            logger.error(f"Error compressing video: {str(e)}")
    
    def extract_frames_at_intervals(self, video_path: str, interval_seconds: float) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames at specified time intervals
        
        Args:
            video_path: Path to video file
            interval_seconds: Time interval between extracted frames
            
        Returns:
            List of (timestamp, frame) tuples
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frames_with_timestamps = []
            frame_interval = int(fps * interval_seconds)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frames_with_timestamps.append((timestamp, frame))
                
                frame_count += 1
            
            cap.release()
            return frames_with_timestamps
            
        except Exception as e:
            logger.error(f"Error extracting frames at intervals: {str(e)}")
            return []
    
    def resize_video(self, frames: List[np.ndarray], target_size: Tuple[int, int], 
                     interpolation: int = cv2.INTER_LINEAR) -> List[np.ndarray]:
        """
        Resize video frames to target size
        
        Args:
            frames: List of video frames
            target_size: Target size as (width, height)
            interpolation: Interpolation method
            
        Returns:
            List of resized frames
        """
        resized_frames = []
        
        for frame in frames:
            resized_frame = cv2.resize(frame, target_size, interpolation=interpolation)
            resized_frames.append(resized_frame)
        
        return resized_frames
    
    def crop_video(self, frames: List[np.ndarray], crop_box: Tuple[int, int, int, int]) -> List[np.ndarray]:
        """
        Crop video frames
        
        Args:
            frames: List of video frames
            crop_box: Crop box as (x, y, width, height)
            
        Returns:
            List of cropped frames
        """
        x, y, w, h = crop_box
        cropped_frames = []
        
        for frame in frames:
            cropped_frame = frame[y:y+h, x:x+w]
            cropped_frames.append(cropped_frame)
        
        return cropped_frames
    
    def stabilize_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply video stabilization to reduce camera shake
        
        Args:
            frames: List of video frames
            
        Returns:
            List of stabilized frames
        """
        if len(frames) < 2:
            return frames
        
        # Convert first frame to grayscale
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Initialize transformation array
        transforms = []
        
        # Calculate transformations between consecutive frames
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Detect feature points
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                             minDistance=30, blockSize=3)
            
            if prev_pts is not None:
                # Calculate optical flow
                curr_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
                
                # Filter good points
                good_prev = prev_pts[status == 1]
                good_curr = curr_pts[status == 1]
                
                if len(good_prev) > 10:
                    # Estimate transformation matrix
                    transform = cv2.estimateAffinePartial2D(good_prev, good_curr)[0]
                    
                    if transform is not None:
                        # Extract transformation parameters
                        dx = transform[0, 2]
                        dy = transform[1, 2]
                        da = np.arctan2(transform[1, 0], transform[0, 0])
                        transforms.append([dx, dy, da])
                    else:
                        transforms.append([0, 0, 0])
                else:
                    transforms.append([0, 0, 0])
            else:
                transforms.append([0, 0, 0])
            
            prev_gray = curr_gray
        
        # Calculate cumulative transformations
        transforms = np.array(transforms)
        trajectory = np.cumsum(transforms, axis=0)
        
        # Smooth trajectory using moving average
        smoothed_trajectory = self._smooth_trajectory(trajectory)
        
        # Calculate smooth transformations
        smooth_transforms = smoothed_trajectory - trajectory
        smooth_transforms = np.vstack([[0, 0, 0], smooth_transforms])
        
        # Apply stabilization
        stabilized_frames = []
        stabilized_frames.append(frames[0])  # First frame unchanged
        
        for i in range(1, len(frames)):
            dx, dy, da = smooth_transforms[i]
            
            # Create transformation matrix
            transform_matrix = np.array([
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da), np.cos(da), dy]
            ], dtype=np.float32)
            
            # Apply transformation
            h, w = frames[i].shape[:2]
            stabilized_frame = cv2.warpAffine(frames[i], transform_matrix, (w, h))
            stabilized_frames.append(stabilized_frame)
        
        return stabilized_frames
    
    def _smooth_trajectory(self, trajectory: np.ndarray, smoothing_radius: int = 30) -> np.ndarray:
        """Smooth trajectory using moving average"""
        smoothed = np.copy(trajectory)
        
        for i in range(trajectory.shape[1]):
            for j in range(len(trajectory)):
                start = max(0, j - smoothing_radius)
                end = min(len(trajectory), j + smoothing_radius + 1)
                smoothed[j, i] = np.mean(trajectory[start:end, i])
        
        return smoothed
    
    def enhance_video_quality(self, frames: List[np.ndarray], enhancement_type: str = 'general') -> List[np.ndarray]:
        """
        Enhance video quality using various techniques
        
        Args:
            frames: List of video frames
            enhancement_type: Type of enhancement ('general', 'denoise', 'sharpen', 'contrast')
            
        Returns:
            List of enhanced frames
        """
        enhanced_frames = []
        
        for frame in frames:
            if enhancement_type == 'general':
                enhanced_frame = self._general_enhancement(frame)
            elif enhancement_type == 'denoise':
                enhanced_frame = self._denoise_frame(frame)
            elif enhancement_type == 'sharpen':
                enhanced_frame = self._sharpen_frame(frame)
            elif enhancement_type == 'contrast':
                enhanced_frame = self._enhance_contrast(frame)
            else:
                enhanced_frame = frame
            
            enhanced_frames.append(enhanced_frame)
        
        return enhanced_frames
    
    def _general_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """Apply general enhancement to frame"""
        # Apply bilateral filter for noise reduction while preserving edges
        enhanced = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply denoising to frame"""
        return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    def _sharpen_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply sharpening to frame"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        
        # Blend with original for subtle effect
        alpha = 0.6
        result = cv2.addWeighted(frame, 1 - alpha, sharpened, alpha, 0)
        
        return result
    
    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Enhance contrast of frame"""
        # Convert to YUV
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Apply histogram equalization to Y channel
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return enhanced
    
    def create_slow_motion(self, frames: List[np.ndarray], slow_factor: float = 2.0) -> List[np.ndarray]:
        """
        Create slow motion effect by interpolating frames
        
        Args:
            frames: List of video frames
            slow_factor: Slow motion factor (2.0 = half speed)
            
        Returns:
            List of frames with slow motion effect
        """
        if slow_factor <= 1.0:
            return frames
        
        slow_frames = []
        
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            slow_frames.append(current_frame)
            
            # Create interpolated frames
            num_interpolated = int(slow_factor) - 1
            
            for j in range(1, num_interpolated + 1):
                alpha = j / (num_interpolated + 1)
                interpolated = cv2.addWeighted(current_frame, 1 - alpha, next_frame, alpha, 0)
                slow_frames.append(interpolated)
        
        # Add last frame
        slow_frames.append(frames[-1])
        
        return slow_frames
    
    def create_time_lapse(self, frames: List[np.ndarray], speed_factor: int = 4) -> List[np.ndarray]:
        """
        Create time-lapse effect by skipping frames
        
        Args:
            frames: List of video frames
            speed_factor: Speed factor (4 = 4x faster)
            
        Returns:
            List of frames with time-lapse effect
        """
        return frames[::speed_factor]
    
    def add_motion_blur(self, frames: List[np.ndarray], blur_strength: int = 15) -> List[np.ndarray]:
        """
        Add motion blur effect to simulate fast movement
        
        Args:
            frames: List of video frames
            blur_strength: Strength of motion blur
            
        Returns:
            List of frames with motion blur
        """
        blurred_frames = []
        
        # Create motion blur kernel
        kernel = np.ones((blur_strength, blur_strength), np.float32) / (blur_strength * blur_strength)
        
        for frame in frames:
            blurred_frame = cv2.filter2D(frame, -1, kernel)
            blurred_frames.append(blurred_frame)
        
        return blurred_frames
    
    def apply_color_grading(self, frames: List[np.ndarray], style: str = 'cinematic') -> List[np.ndarray]:
        """
        Apply color grading to video
        
        Args:
            frames: List of video frames
            style: Color grading style ('cinematic', 'vintage', 'cold', 'warm')
            
        Returns:
            List of color graded frames
        """
        graded_frames = []
        
        for frame in frames:
            if style == 'cinematic':
                graded_frame = self._apply_cinematic_look(frame)
            elif style == 'vintage':
                graded_frame = self._apply_vintage_look(frame)
            elif style == 'cold':
                graded_frame = self._apply_cold_look(frame)
            elif style == 'warm':
                graded_frame = self._apply_warm_look(frame)
            else:
                graded_frame = frame
            
            graded_frames.append(graded_frame)
        
        return graded_frames
    
    def _apply_cinematic_look(self, frame: np.ndarray) -> np.ndarray:
        """Apply cinematic color grading"""
        # Increase contrast and reduce saturation slightly
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Reduce saturation
        a = cv2.addWeighted(a, 0.8, np.zeros_like(a), 0.2, 0)
        b = cv2.addWeighted(b, 0.8, np.zeros_like(b), 0.2, 0)
        
        graded = cv2.merge([l, a, b])
        graded = cv2.cvtColor(graded, cv2.COLOR_LAB2BGR)
        
        return graded
    
    def _apply_vintage_look(self, frame: np.ndarray) -> np.ndarray:
        """Apply vintage color grading"""
        # Add sepia tone effect
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        
        vintage = cv2.transform(frame, kernel)
        vintage = np.clip(vintage, 0, 255).astype(np.uint8)
        
        # Add slight vignette
        h, w = frame.shape[:2]
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        center_x, center_y = w // 2, h // 2
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        vignette = 1 - (distance / max_distance) * 0.3
        vignette = np.clip(vignette, 0, 1)
        vignette = np.repeat(vignette[:, :, np.newaxis], 3, axis=2)
        
        vintage = (vintage * vignette).astype(np.uint8)
        
        return vintage
    
    def _apply_cold_look(self, frame: np.ndarray) -> np.ndarray:
        """Apply cold color grading (blue tint)"""
        # Increase blue channel, decrease red
        cold = frame.copy().astype(np.float32)
        cold[:, :, 0] = np.clip(cold[:, :, 0] * 1.2, 0, 255)  # Blue
        cold[:, :, 2] = np.clip(cold[:, :, 2] * 0.8, 0, 255)  # Red
        
        return cold.astype(np.uint8)
    
    def _apply_warm_look(self, frame: np.ndarray) -> np.ndarray:
        """Apply warm color grading (orange tint)"""
        # Increase red and green channels, decrease blue
        warm = frame.copy().astype(np.float32)
        warm[:, :, 0] = np.clip(warm[:, :, 0] * 0.8, 0, 255)  # Blue
        warm[:, :, 1] = np.clip(warm[:, :, 1] * 1.1, 0, 255)  # Green
        warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.2, 0, 255)  # Red
        
        return warm.astype(np.uint8)
    
    def create_video_mosaic(self, video_paths: List[str], output_path: str, 
                           grid_size: Tuple[int, int] = (2, 2)) -> None:
        """
        Create mosaic video from multiple input videos
        
        Args:
            video_paths: List of paths to input videos
            output_path: Path for output mosaic video
            grid_size: Grid size as (rows, cols)
        """
        try:
            # Load all videos
            video_clips = []
            min_duration = float('inf')
            
            for path in video_paths:
                clip = VideoFileClip(path)
                video_clips.append(clip)
                min_duration = min(min_duration, clip.duration)
            
            # Trim all clips to minimum duration
            video_clips = [clip.subclip(0, min_duration) for clip in video_clips]
            
            # Resize clips to fit in grid
            rows, cols = grid_size
            target_width = video_clips[0].w // cols
            target_height = video_clips[0].h // rows
            
            resized_clips = [clip.resize((target_width, target_height)) for clip in video_clips]
            
            # Arrange clips in grid
            rows_of_clips = []
            for i in range(rows):
                row_clips = []
                for j in range(cols):
                    idx = i * cols + j
                    if idx < len(resized_clips):
                        row_clips.append(resized_clips[idx])
                    else:
                        # Create black clip if not enough videos
                        black_clip = resized_clips[0].fx(lambda gf, t: np.zeros_like(gf(0)))
                        row_clips.append(black_clip)
                
                row_clip = concatenate_videoclips(row_clips, method='compose')
                rows_of_clips.append(row_clip)
            
            # Combine rows
            from moviepy.editor import concatenate_videoclips
            final_clip = concatenate_videoclips(rows_of_clips, method='compose')
            
            # Write output
            final_clip.write_videofile(output_path, verbose=False, logger=None)
            
            # Clean up
            for clip in video_clips + resized_clips:
                clip.close()
            final_clip.close()
            
        except Exception as e:
            logger.error(f"Error creating video mosaic: {str(e)}")
            raise
    
    def extract_audio(self, video_path: str, output_audio_path: str) -> None:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to input video
            output_audio_path: Path for output audio file
        """
        try:
            with VideoFileClip(video_path) as video_clip:
                if video_clip.audio is not None:
                    video_clip.audio.write_audiofile(output_audio_path, verbose=False, logger=None)
                    logger.info(f"Audio extracted to: {output_audio_path}")
                else:
                    logger.warning("No audio track found in video")
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get comprehensive information about video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            info = {
                'path': video_path,
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0,
                'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
                'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
            }
            
            if info['fps'] > 0:
                info['duration'] = info['frame_count'] / info['fps']
            
            cap.release()
            
            # Get additional info using moviepy
            try:
                with VideoFileClip(video_path) as clip:
                    info['has_audio'] = clip.audio is not None
                    if clip.audio:
                        info['audio_fps'] = clip.audio.fps
                        info['audio_duration'] = clip.audio.duration
            except:
                info['has_audio'] = False
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {}
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {str(e)}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_temp_files()