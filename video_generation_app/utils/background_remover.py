import cv2
import numpy as np
from typing import Optional, Tuple, Union
import logging
from pathlib import Path
import os

# Try to import rembg for advanced background removal
try:
    from rembg import remove as rembg_remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg not available, using basic background removal")

logger = logging.getLogger(__name__)

class BackgroundRemover:
    """Advanced background removal and replacement system"""
    
    def __init__(self):
        """Initialize background remover"""
        self.rembg_session = None
        
        if REMBG_AVAILABLE:
            try:
                self.rembg_session = new_session('u2net')
                logger.info("Initialized rembg with u2net model")
            except Exception as e:
                logger.warning(f"Failed to initialize rembg: {str(e)}")
                self.rembg_session = None
        
        # Background subtractor for simple cases
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            history=500,
            varThreshold=50
        )
        
        # Edge detection parameters
        self.edge_params = {
            'canny_low': 50,
            'canny_high': 150,
            'blur_kernel': (5, 5),
            'morphology_kernel': np.ones((3, 3), np.uint8)
        }
        
        # Matting parameters
        self.alpha_matting_params = {
            'trimap_erosion': 10,
            'trimap_dilation': 20,
            'unknown_region_size': 30
        }
    
    def replace_background(self, frame: np.ndarray, background_path: str) -> np.ndarray:
        """
        Replace background in frame with new background
        
        Args:
            frame: Input frame
            background_path: Path to new background image
            
        Returns:
            Frame with replaced background
        """
        try:
            # Load new background
            new_background = cv2.imread(background_path)
            if new_background is None:
                logger.error(f"Could not load background image: {background_path}")
                return frame
            
            # Resize background to match frame
            new_background = cv2.resize(new_background, (frame.shape[1], frame.shape[0]))
            
            # Remove original background
            foreground_mask = self.remove_background(frame)
            
            # Create alpha mask
            alpha_mask = self._create_alpha_mask(foreground_mask)
            
            # Blend foreground with new background
            result = self._blend_with_background(frame, new_background, alpha_mask)
            
            # Post-process for better quality
            result = self._post_process_background(result, alpha_mask)
            
            return result
            
        except Exception as e:
            logger.error(f"Error replacing background: {str(e)}")
            return frame
    
    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        """
        Remove background from frame
        
        Args:
            frame: Input frame
            
        Returns:
            Foreground mask (0-255)
        """
        if self.rembg_session is not None:
            return self._remove_background_rembg(frame)
        else:
            return self._remove_background_basic(frame)
    
    def _remove_background_rembg(self, frame: np.ndarray) -> np.ndarray:
        """Remove background using rembg library"""
        try:
            # Convert frame format for rembg
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Remove background
            result = rembg_remove(frame_rgb, session=self.rembg_session)
            
            # Extract alpha channel as mask
            if result.shape[2] == 4:  # RGBA
                mask = result[:, :, 3]
            else:  # RGB (fallback)
                mask = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error in rembg background removal: {str(e)}")
            return self._remove_background_basic(frame)
    
    def _remove_background_basic(self, frame: np.ndarray) -> np.ndarray:
        """Basic background removal using traditional CV methods"""
        # Method 1: Color-based segmentation
        mask1 = self._color_based_segmentation(frame)
        
        # Method 2: Edge-based segmentation
        mask2 = self._edge_based_segmentation(frame)
        
        # Method 3: GrabCut algorithm
        mask3 = self._grabcut_segmentation(frame)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask1, mask2)
        combined_mask = cv2.bitwise_or(combined_mask, mask3)
        
        # Refine mask
        refined_mask = self._refine_mask(combined_mask, frame)
        
        return refined_mask
    
    def _color_based_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """Segment foreground using color clustering"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Focus on center region (likely to contain person)
        h, w = frame.shape[:2]
        center_region = frame[h//4:3*h//4, w//4:3*w//4]
        
        # Get dominant colors in center region
        center_colors = center_region.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(center_colors)
        
        # Create mask based on dominant colors
        distances = np.linalg.norm(frame.reshape(-1, 3)[:, None] - kmeans.cluster_centers_, axis=2)
        closest_cluster = np.argmin(distances, axis=1)
        
        # Assume the most common cluster in center is foreground
        center_flat = center_region.reshape(-1, 3)
        center_distances = np.linalg.norm(center_flat[:, None] - kmeans.cluster_centers_, axis=2)
        center_clusters = np.argmin(center_distances, axis=1)
        foreground_cluster = np.bincount(center_clusters).argmax()
        
        # Create foreground mask
        mask = (closest_cluster == foreground_cluster).reshape(h, w).astype(np.uint8) * 255
        
        return mask
    
    def _edge_based_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """Segment foreground using edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.edge_params['blur_kernel'], 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.edge_params['canny_low'], self.edge_params['canny_high'])
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask from largest contours
        mask = np.zeros_like(gray)
        
        if contours:
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Fill largest contours (likely to be person)
            for i, contour in enumerate(contours[:3]):  # Top 3 contours
                if cv2.contourArea(contour) > frame.size * 0.01:  # Minimum area threshold
                    cv2.fillPoly(mask, [contour], 255)
        
        return mask
    
    def _grabcut_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """Segment foreground using GrabCut algorithm"""
        try:
            # Create initial mask
            mask = np.zeros(frame.shape[:2], np.uint8)
            
            # Define rectangle around center region
            h, w = frame.shape[:2]
            rect = (w//6, h//6, 2*w//3, 2*h//3)
            
            # Initialize background and foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create final mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result_mask = mask2 * 255
            
            return result_mask
            
        except Exception as e:
            logger.warning(f"GrabCut segmentation failed: {str(e)}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)
    
    def _refine_mask(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Refine segmentation mask"""
        # Apply morphological operations
        kernel = self.edge_params['morphology_kernel']
        
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Apply threshold to maintain binary mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _create_alpha_mask(self, foreground_mask: np.ndarray) -> np.ndarray:
        """Create smooth alpha mask for blending"""
        # Apply Gaussian blur for smooth edges
        alpha_mask = cv2.GaussianBlur(foreground_mask, (15, 15), 0)
        
        # Normalize to 0-1 range
        alpha_mask = alpha_mask.astype(np.float32) / 255.0
        
        # Apply alpha matting for better edges
        alpha_mask = self._apply_alpha_matting(alpha_mask, foreground_mask)
        
        return alpha_mask
    
    def _apply_alpha_matting(self, alpha_mask: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """Apply alpha matting for smoother edges"""
        # Create trimap
        trimap = self._create_trimap(binary_mask)
        
        # Simple alpha matting using distance transform
        # Find unknown regions
        unknown_region = (trimap == 128)
        
        if np.any(unknown_region):
            # Distance transform from foreground
            fg_dist = cv2.distanceTransform((trimap == 255).astype(np.uint8), cv2.DIST_L2, 5)
            
            # Distance transform from background
            bg_dist = cv2.distanceTransform((trimap == 0).astype(np.uint8), cv2.DIST_L2, 5)
            
            # Calculate alpha values in unknown region
            total_dist = fg_dist + bg_dist
            alpha_unknown = np.divide(fg_dist, total_dist, out=np.zeros_like(fg_dist), where=total_dist!=0)
            
            # Update alpha mask
            alpha_mask[unknown_region] = alpha_unknown[unknown_region]
        
        return alpha_mask
    
    def _create_trimap(self, binary_mask: np.ndarray) -> np.ndarray:
        """Create trimap for alpha matting"""
        # Erode for sure foreground
        kernel = np.ones((self.alpha_matting_params['trimap_erosion'], 
                         self.alpha_matting_params['trimap_erosion']), np.uint8)
        sure_fg = cv2.erode(binary_mask, kernel, iterations=1)
        
        # Dilate for sure background
        kernel = np.ones((self.alpha_matting_params['trimap_dilation'], 
                         self.alpha_matting_params['trimap_dilation']), np.uint8)
        sure_bg_inv = cv2.dilate(binary_mask, kernel, iterations=1)
        sure_bg = cv2.bitwise_not(sure_bg_inv)
        
        # Create trimap
        trimap = np.zeros_like(binary_mask)
        trimap[sure_bg == 255] = 0      # Background
        trimap[sure_fg == 255] = 255    # Foreground
        trimap[(sure_bg != 255) & (sure_fg != 255)] = 128  # Unknown
        
        return trimap
    
    def _blend_with_background(self, foreground: np.ndarray, background: np.ndarray, 
                              alpha_mask: np.ndarray) -> np.ndarray:
        """Blend foreground with new background using alpha mask"""
        # Ensure alpha mask has 3 channels
        if len(alpha_mask.shape) == 2:
            alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
        
        # Blend images
        result = foreground.astype(np.float32) * alpha_mask + background.astype(np.float32) * (1 - alpha_mask)
        
        return result.astype(np.uint8)
    
    def _post_process_background(self, result: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
        """Post-process the result for better quality"""
        # Edge enhancement around the subject
        if len(alpha_mask.shape) == 3:
            alpha_gray = cv2.cvtColor(alpha_mask, cv2.COLOR_BGR2GRAY)
        else:
            alpha_gray = alpha_mask
        
        # Find edges of the alpha mask
        edges = cv2.Canny((alpha_gray * 255).astype(np.uint8), 50, 150)
        
        # Dilate edges slightly
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply slight sharpening to edges
        edge_mask = edges.astype(np.float32) / 255.0
        edge_mask = np.repeat(edge_mask[:, :, np.newaxis], 3, axis=2)
        
        # Sharpen edges
        sharpened = cv2.filter2D(result, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
        result = result.astype(np.float32) * (1 - edge_mask) + sharpened.astype(np.float32) * edge_mask
        
        return result.astype(np.uint8)
    
    def create_green_screen_mask(self, frame: np.ndarray, 
                                color_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Create mask for green screen (chroma key) removal
        
        Args:
            frame: Input frame
            color_range: Tuple of (lower_bound, upper_bound) in HSV format
            
        Returns:
            Mask for non-green screen areas
        """
        if color_range is None:
            # Default green screen range in HSV
            lower_green = (40, 50, 50)
            upper_green = (80, 255, 255)
        else:
            lower_green, upper_green = color_range
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green screen
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Invert mask (we want non-green areas)
        foreground_mask = cv2.bitwise_not(green_mask)
        
        # Refine mask
        foreground_mask = self._refine_mask(foreground_mask, frame)
        
        return foreground_mask
    
    def create_blue_screen_mask(self, frame: np.ndarray,
                               color_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Create mask for blue screen (chroma key) removal
        
        Args:
            frame: Input frame
            color_range: Tuple of (lower_bound, upper_bound) in HSV format
            
        Returns:
            Mask for non-blue screen areas
        """
        if color_range is None:
            # Default blue screen range in HSV
            lower_blue = (100, 50, 50)
            upper_blue = (130, 255, 255)
        else:
            lower_blue, upper_blue = color_range
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue screen
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Invert mask (we want non-blue areas)
        foreground_mask = cv2.bitwise_not(blue_mask)
        
        # Refine mask
        foreground_mask = self._refine_mask(foreground_mask, frame)
        
        return foreground_mask
    
    def apply_background_blur(self, frame: np.ndarray, blur_strength: int = 15) -> np.ndarray:
        """
        Apply background blur effect (bokeh)
        
        Args:
            frame: Input frame
            blur_strength: Strength of blur effect
            
        Returns:
            Frame with blurred background
        """
        # Get foreground mask
        foreground_mask = self.remove_background(frame)
        
        # Create alpha mask
        alpha_mask = self._create_alpha_mask(foreground_mask)
        
        # Create blurred background
        blurred_background = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        
        # Blend sharp foreground with blurred background
        result = self._blend_with_background(frame, blurred_background, alpha_mask)
        
        return result
    
    def add_background_effects(self, frame: np.ndarray, effect_type: str = "blur") -> np.ndarray:
        """
        Add various background effects
        
        Args:
            frame: Input frame
            effect_type: Type of effect ('blur', 'artistic', 'vintage', 'black_white')
            
        Returns:
            Frame with background effect applied
        """
        # Get foreground mask
        foreground_mask = self.remove_background(frame)
        alpha_mask = self._create_alpha_mask(foreground_mask)
        
        # Apply effect to background
        if effect_type == "blur":
            background = cv2.GaussianBlur(frame, (21, 21), 0)
        elif effect_type == "artistic":
            background = cv2.bilateralFilter(frame, 15, 80, 80)
            background = cv2.edgePreservingFilter(background, flags=1, sigma_s=50, sigma_r=0.4)
        elif effect_type == "vintage":
            background = cv2.applyColorMap(frame, cv2.COLORMAP_AUTUMN)
        elif effect_type == "black_white":
            background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        else:
            background = frame
        
        # Blend foreground with processed background
        result = self._blend_with_background(frame, background, alpha_mask)
        
        return result
    
    def replace_with_video_background(self, frame: np.ndarray, background_frame: np.ndarray) -> np.ndarray:
        """
        Replace background with frame from background video
        
        Args:
            frame: Input frame
            background_frame: Frame from background video
            
        Returns:
            Frame with video background
        """
        # Resize background frame to match input frame
        background_frame = cv2.resize(background_frame, (frame.shape[1], frame.shape[0]))
        
        # Use regular background replacement
        return self.replace_background_with_frame(frame, background_frame)
    
    def replace_background_with_frame(self, frame: np.ndarray, background_frame: np.ndarray) -> np.ndarray:
        """
        Replace background with another frame
        
        Args:
            frame: Input frame
            background_frame: Background frame
            
        Returns:
            Frame with replaced background
        """
        # Remove original background
        foreground_mask = self.remove_background(frame)
        
        # Create alpha mask
        alpha_mask = self._create_alpha_mask(foreground_mask)
        
        # Blend with new background
        result = self._blend_with_background(frame, background_frame, alpha_mask)
        
        # Post-process
        result = self._post_process_background(result, alpha_mask)
        
        return result
    
    def get_background_statistics(self, frame: np.ndarray) -> dict:
        """
        Get statistics about the background
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with background statistics
        """
        foreground_mask = self.remove_background(frame)
        background_mask = cv2.bitwise_not(foreground_mask)
        
        # Calculate background area
        total_pixels = frame.shape[0] * frame.shape[1]
        background_pixels = np.sum(background_mask > 0)
        foreground_pixels = np.sum(foreground_mask > 0)
        
        # Calculate average colors
        background_region = frame[background_mask > 0]
        foreground_region = frame[foreground_mask > 0]
        
        stats = {
            'background_ratio': background_pixels / total_pixels,
            'foreground_ratio': foreground_pixels / total_pixels,
            'background_avg_color': np.mean(background_region, axis=0) if len(background_region) > 0 else [0, 0, 0],
            'foreground_avg_color': np.mean(foreground_region, axis=0) if len(foreground_region) > 0 else [0, 0, 0],
            'complexity_score': self._calculate_background_complexity(frame, background_mask)
        }
        
        return stats
    
    def _calculate_background_complexity(self, frame: np.ndarray, background_mask: np.ndarray) -> float:
        """Calculate complexity score of background"""
        # Extract background region
        background_region = frame.copy()
        background_region[background_mask == 0] = 0
        
        # Calculate gradient magnitude
        gray = cv2.cvtColor(background_region, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate complexity as average gradient in background region
        background_gradients = gradient_magnitude[background_mask > 0]
        complexity = np.mean(background_gradients) if len(background_gradients) > 0 else 0
        
        return float(complexity)