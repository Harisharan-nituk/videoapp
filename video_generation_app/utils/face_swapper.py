import cv2
import numpy as np
import face_recognition
import dlib
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class FaceSwapper:
    """Advanced face swapping using face recognition and dlib"""
    
    def __init__(self, face_model=None):
        """Initialize face swapper"""
        self.face_model = face_model
        
        # Initialize dlib face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Download predictor if not exists
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            logger.warning("Face landmarks predictor not found. Download from dlib website.")
            self.predictor = None
        else:
            self.predictor = dlib.shape_predictor(predictor_path)
        
        # Face alignment parameters
        self.face_align = True
        self.blend_mode = "seamless"
        
        # Cache for face encodings
        self.face_cache = {}
    
    def swap_face(self, frame: np.ndarray, face_image_path: str, pose_data: Dict = None) -> np.ndarray:
        """
        Swap face in frame with provided face image
        
        Args:
            frame: Input frame
            face_image_path: Path to replacement face image
            pose_data: Optional pose data for better alignment
            
        Returns:
            Frame with face swapped
        """
        try:
            # Load replacement face
            replacement_face = cv2.imread(face_image_path)
            if replacement_face is None:
                logger.error(f"Could not load face image: {face_image_path}")
                return frame
            
            # Detect faces in both images
            frame_faces = self._detect_faces(frame)
            replacement_faces = self._detect_faces(replacement_face)
            
            if not frame_faces or not replacement_faces:
                logger.warning("No faces detected for swapping")
                return frame
            
            # Get the best face from replacement image
            best_replacement_face = self._get_best_face(replacement_faces, replacement_face)
            
            # Swap each detected face in the frame
            result_frame = frame.copy()
            for face_data in frame_faces:
                result_frame = self._perform_face_swap(
                    result_frame, 
                    face_data, 
                    replacement_face, 
                    best_replacement_face
                )
            
            return result_frame
            
        except Exception as e:
            logger.error(f"Error in face swapping: {str(e)}")
            return frame
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces and extract face data
        
        Args:
            image: Input image
            
        Returns:
            List of face data dictionaries
        """
        faces = []
        
        # Convert to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        # Extract face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Extract face landmarks using dlib if available
        if self.predictor is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dlib_faces = self.detector(gray)
            
            for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
                top, right, bottom, left = location
                
                # Find corresponding dlib face
                dlib_face = None
                for df in dlib_faces:
                    if (abs(df.left() - left) < 20 and 
                        abs(df.top() - top) < 20 and
                        abs(df.right() - right) < 20 and
                        abs(df.bottom() - bottom) < 20):
                        dlib_face = df
                        break
                
                # Extract landmarks
                landmarks = None
                if dlib_face is not None:
                    landmarks = self.predictor(gray, dlib_face)
                    landmarks = self._shape_to_np(landmarks)
                
                face_data = {
                    'encoding': encoding,
                    'location': location,
                    'landmarks': landmarks,
                    'bbox': (left, top, right, bottom),
                    'center': ((left + right) // 2, (top + bottom) // 2),
                    'size': (right - left, bottom - top)
                }
                
                faces.append(face_data)
        else:
            # Fallback without landmarks
            for encoding, location in zip(face_encodings, face_locations):
                top, right, bottom, left = location
                face_data = {
                    'encoding': encoding,
                    'location': location,
                    'landmarks': None,
                    'bbox': (left, top, right, bottom),
                    'center': ((left + right) // 2, (top + bottom) // 2),
                    'size': (right - left, bottom - top)
                }
                faces.append(face_data)
        
        return faces
    
    def _shape_to_np(self, shape, dtype="int"):
        """Convert dlib shape to numpy array"""
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def _get_best_face(self, faces: List[Dict], image: np.ndarray) -> Dict:
        """
        Select the best face from detected faces
        
        Args:
            faces: List of detected faces
            image: Source image
            
        Returns:
            Best face data
        """
        if len(faces) == 1:
            return faces[0]
        
        # Score faces based on size and quality
        best_face = faces[0]
        best_score = 0
        
        for face in faces:
            # Calculate face size score
            width, height = face['size']
            size_score = width * height
            
            # Calculate face quality score (based on landmarks if available)
            quality_score = 1.0
            if face['landmarks'] is not None:
                # Check if landmarks are well-defined
                landmarks = face['landmarks']
                landmark_variance = np.var(landmarks, axis=0)
                quality_score = np.mean(landmark_variance) / 1000.0
            
            # Combine scores
            total_score = size_score * quality_score
            
            if total_score > best_score:
                best_score = total_score
                best_face = face
        
        return best_face
    
    def _perform_face_swap(self, frame: np.ndarray, target_face: Dict, 
                          source_image: np.ndarray, source_face: Dict) -> np.ndarray:
        """
        Perform the actual face swap
        
        Args:
            frame: Target frame
            target_face: Target face data
            source_image: Source image containing replacement face
            source_face: Source face data
            
        Returns:
            Frame with swapped face
        """
        if target_face['landmarks'] is None or source_face['landmarks'] is None:
            return self._simple_face_swap(frame, target_face, source_image, source_face)
        
        # Extract face regions
        target_landmarks = target_face['landmarks']
        source_landmarks = source_face['landmarks']
        
        # Get face masks
        target_mask = self._get_face_mask(frame.shape, target_landmarks)
        source_mask = self._get_face_mask(source_image.shape, source_landmarks)
        
        # Align source face to target face
        aligned_source = self._align_face(
            source_image, 
            source_landmarks, 
            target_landmarks,
            frame.shape
        )
        
        # Blend the faces
        if self.blend_mode == "seamless":
            result = self._seamless_blend(frame, aligned_source, target_mask)
        else:
            result = self._alpha_blend(frame, aligned_source, target_mask)
        
        return result
    
    def _simple_face_swap(self, frame: np.ndarray, target_face: Dict, 
                         source_image: np.ndarray, source_face: Dict) -> np.ndarray:
        """
        Simple face swap without landmarks
        
        Args:
            frame: Target frame
            target_face: Target face data
            source_image: Source image
            source_face: Source face data
            
        Returns:
            Frame with swapped face
        """
        # Get face regions
        target_bbox = target_face['bbox']
        source_bbox = source_face['bbox']
        
        # Extract faces
        t_left, t_top, t_right, t_bottom = target_bbox
        s_left, s_top, s_right, s_bottom = source_bbox
        
        source_face_img = source_image[s_top:s_bottom, s_left:s_right]
        
        # Resize source face to match target
        target_width = t_right - t_left
        target_height = t_bottom - t_top
        
        resized_source = cv2.resize(source_face_img, (target_width, target_height))
        
        # Create simple mask
        mask = np.ones((target_height, target_width, 3), dtype=np.float32)
        
        # Apply Gaussian blur to mask edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Blend faces
        result = frame.copy()
        roi = result[t_top:t_bottom, t_left:t_right]
        blended = roi * (1 - mask) + resized_source * mask
        result[t_top:t_bottom, t_left:t_right] = blended.astype(np.uint8)
        
        return result
    
    def _get_face_mask(self, image_shape: Tuple, landmarks: np.ndarray) -> np.ndarray:
        """
        Create face mask from landmarks
        
        Args:
            image_shape: Shape of target image
            landmarks: Face landmarks
            
        Returns:
            Face mask
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Use face boundary landmarks (0-16 for jaw, 17-26 for eyebrows)
        face_boundary = np.concatenate([
            landmarks[0:17],  # Jaw line
            landmarks[26:16:-1]  # Eyebrow to eyebrow (reversed)
        ])
        
        # Create convex hull
        hull = cv2.convexHull(face_boundary.astype(np.int32))
        cv2.fillPoly(mask, [hull], 255)
        
        # Apply Gaussian blur for smooth edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Convert to 3-channel mask
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = mask.astype(np.float32) / 255.0
        
        return mask
    
    def _align_face(self, source_image: np.ndarray, source_landmarks: np.ndarray,
                   target_landmarks: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """
        Align source face to target face using landmarks
        
        Args:
            source_image: Source image
            source_landmarks: Source face landmarks
            target_landmarks: Target face landmarks
            target_shape: Target image shape
            
        Returns:
            Aligned source face
        """
        # Select key points for alignment (eyes, nose, mouth)
        key_points_indices = [36, 45, 33, 48, 54]  # Left eye, right eye, nose, mouth corners
        
        source_points = source_landmarks[key_points_indices].astype(np.float32)
        target_points = target_landmarks[key_points_indices].astype(np.float32)
        
        # Calculate transformation matrix
        transform_matrix = cv2.estimateAffinePartial2D(source_points, target_points)[0]
        
        if transform_matrix is None:
            # Fallback to simple scaling and translation
            source_center = np.mean(source_landmarks, axis=0)
            target_center = np.mean(target_landmarks, axis=0)
            
            # Calculate scale
            source_scale = np.std(source_landmarks, axis=0)
            target_scale = np.std(target_landmarks, axis=0)
            scale = np.mean(target_scale / source_scale)
            
            # Create transformation matrix
            transform_matrix = np.array([
                [scale, 0, target_center[0] - source_center[0] * scale],
                [0, scale, target_center[1] - source_center[1] * scale]
            ], dtype=np.float32)
        
        # Apply transformation
        aligned_image = cv2.warpAffine(
            source_image, 
            transform_matrix, 
            (target_shape[1], target_shape[0])
        )
        
        return aligned_image
    
    def _seamless_blend(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Seamless blending using Poisson blending
        
        Args:
            target: Target image
            source: Source image
            mask: Blend mask
            
        Returns:
            Blended image
        """
        try:
            # Convert mask to single channel
            if len(mask.shape) == 3:
                mask_single = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_single = mask
            
            # Find center of mask
            moments = cv2.moments(mask_single)
            if moments['m00'] != 0:
                center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            else:
                center = (target.shape[1] // 2, target.shape[0] // 2)
            
            # Apply seamless cloning
            result = cv2.seamlessClone(
                source, 
                target, 
                mask_single, 
                center, 
                cv2.NORMAL_CLONE
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Seamless blending failed, falling back to alpha blend: {str(e)}")
            return self._alpha_blend(target, source, mask)
    
    def _alpha_blend(self, target: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Alpha blending
        
        Args:
            target: Target image
            source: Source image
            mask: Blend mask
            
        Returns:
            Blended image
        """
        # Ensure mask is float
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        
        # Alpha blend
        result = target.astype(np.float32) * (1 - mask) + source.astype(np.float32) * mask
        
        return result.astype(np.uint8)
    
    def enhance_face_quality(self, face_region: np.ndarray) -> np.ndarray:
        """
        Enhance face quality using image processing
        
        Args:
            face_region: Face region to enhance
            
        Returns:
            Enhanced face region
        """
        # Apply bilateral filter for noise reduction while preserving edges
        enhanced = cv2.bilateralFilter(face_region, 9, 75, 75)
        
        # Enhance contrast
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_face_emotions(self, face_region: np.ndarray) -> Dict[str, float]:
        """
        Detect emotions in face region (placeholder for emotion detection)
        
        Args:
            face_region: Face region
            
        Returns:
            Dictionary of emotion scores
        """
        # Placeholder implementation
        # In a real implementation, you would use an emotion detection model
        emotions = {
            'happy': 0.5,
            'sad': 0.1,
            'angry': 0.1,
            'surprise': 0.1,
            'fear': 0.1,
            'disgust': 0.1,
            'neutral': 0.1
        }
        
        return emotions
    
    def preserve_facial_expressions(self, target_face: Dict, source_face: Dict) -> Dict:
        """
        Preserve facial expressions from target in source face
        
        Args:
            target_face: Target face data
            source_face: Source face data
            
        Returns:
            Modified source face data with preserved expressions
        """
        if target_face['landmarks'] is None or source_face['landmarks'] is None:
            return source_face
        
        # Extract expression-related landmarks
        target_landmarks = target_face['landmarks']
        source_landmarks = source_face['landmarks'].copy()
        
        # Mouth region (48-67)
        mouth_offset = target_landmarks[48:68] - np.mean(target_landmarks[48:68], axis=0)
        source_mouth_center = np.mean(source_landmarks[48:68], axis=0)
        source_landmarks[48:68] = source_mouth_center + mouth_offset
        
        # Eye regions (36-47)
        # Left eye
        left_eye_offset = target_landmarks[36:42] - np.mean(target_landmarks[36:42], axis=0)
        source_left_eye_center = np.mean(source_landmarks[36:42], axis=0)
        source_landmarks[36:42] = source_left_eye_center + left_eye_offset
        
        # Right eye
        right_eye_offset = target_landmarks[42:48] - np.mean(target_landmarks[42:48], axis=0)
        source_right_eye_center = np.mean(source_landmarks[42:48], axis=0)
        source_landmarks[42:48] = source_right_eye_center + right_eye_offset
        
        # Update source face data
        modified_source = source_face.copy()
        modified_source['landmarks'] = source_landmarks
        
        return modified_source