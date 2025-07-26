"""
Advanced Face Swapping Module
Handles face detection, alignment, and seamless blending
"""

import cv2
import numpy as np
import dlib
import face_recognition
from typing import Optional, Tuple

class AdvancedFaceSwapper:
    """Advanced face swapping with better blending and alignment"""
    
    def __init__(self, landmarks_model_path: str = "assets/shape_predictor_68_face_landmarks.dat"):
        """
        Initialize the face swapper
        
        Args:
            landmarks_model_path: Path to the dlib landmarks model
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmarks_model_path)
        self.face_encoder = face_recognition.api
        
    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Landmarks as numpy array or None if no face found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
            
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Convert to numpy array
        points = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)
            
        return points
    
    def align_faces(self, source_face: np.ndarray, target_landmarks: np.ndarray, 
                   source_landmarks: np.ndarray) -> np.ndarray:
        """
        Align source face to target face using landmarks
        
        Args:
            source_face: Source face image
            target_landmarks: Target face landmarks
            source_landmarks: Source face landmarks
            
        Returns:
            Aligned face image
        """
        # Get transformation matrix
        transformation_matrix = cv2.estimateAffinePartial2D(
            source_landmarks, target_landmarks
        )[0]
        
        # Apply transformation
        aligned_face = cv2.warpAffine(
            source_face, transformation_matrix, 
            (source_face.shape[1], source_face.shape[0])
        )
        
        return aligned_face
    
    def create_face_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Create mask for face region
        
        Args:
            landmarks: Face landmarks
            image_shape: Shape of the image
            
        Returns:
            Face mask
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Face boundary points (jawline + forehead)
        face_points = np.concatenate([
            landmarks[0:17],  # Jawline
            landmarks[17:27]  # Eyebrows
        ])
        
        # Create convex hull
        hull = cv2.convexHull(face_points)
        cv2.fillPoly(mask, [hull], 255)
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def seamless_blend(self, target_image: np.ndarray, source_face: np.ndarray, 
                      mask: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Seamlessly blend source face onto target image
        
        Args:
            target_image: Target image
            source_face: Source face image
            mask: Face mask
            landmarks: Face landmarks
            
        Returns:
            Blended image
        """
        # Create seamless clone
        center = tuple(np.mean(landmarks, axis=0).astype(int))
        
        # Use Poisson blending
        result = cv2.seamlessClone(
            source_face, target_image, mask, center, cv2.NORMAL_CLONE
        )
        
        return result
    
    def swap_faces(self, target_image: np.ndarray, source_face_image: np.ndarray) -> np.ndarray:
        """
        Main face swapping function
        
        Args:
            target_image: Image containing target face
            source_face_image: Image containing source face
            
        Returns:
            Image with swapped faces
        """
        target_landmarks = self.get_landmarks(target_image)
        source_landmarks = self.get_landmarks(source_face_image)
        
        if target_landmarks is None or source_landmarks is None:
            return target_image
        
        # Align source face to target
        aligned_source = self.align_faces(source_face_image, target_landmarks, source_landmarks)
        
        # Create mask
        mask = self.create_face_mask(target_landmarks, target_image.shape)
        
        # Blend faces
        result = self.seamless_blend(target_image, aligned_source, mask, target_landmarks)
        
        return result
