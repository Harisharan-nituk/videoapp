import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PoseEstimator:
    """Human pose estimation using MediaPipe"""
    
    def __init__(self, pose_model=None):
        """Initialize pose estimator"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Pose landmarks mapping
        self.pose_landmarks = {
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13,
            'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
            'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24, 'left_knee': 25,
            'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31,
            'right_foot_index': 32
        }
        
        # Store learned patterns
        self.learned_patterns = []
    
    def detect_pose(self, frame: np.ndarray) -> Dict:
        """
        Detect pose in a single frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing pose data
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            pose_data = {
                'landmarks': None,
                'segmentation_mask': None,
                'visibility': None,
                'world_landmarks': None,
                'pose_present': False
            }
            
            if results.pose_landmarks:
                pose_data['pose_present'] = True
                pose_data['landmarks'] = self._extract_landmarks(results.pose_landmarks)
                pose_data['visibility'] = self._extract_visibility(results.pose_landmarks)
                
                if results.segmentation_mask is not None:
                    pose_data['segmentation_mask'] = results.segmentation_mask
                
                if results.pose_world_landmarks:
                    pose_data['world_landmarks'] = self._extract_world_landmarks(
                        results.pose_world_landmarks
                    )
            
            return pose_data
            
        except Exception as e:
            logger.error(f"Error in pose detection: {str(e)}")
            return {'pose_present': False}
    
    def _extract_landmarks(self, landmarks) -> np.ndarray:
        """Extract 2D landmarks coordinates"""
        coords = []
        for landmark in landmarks.landmark:
            coords.append([landmark.x, landmark.y])
        return np.array(coords)
    
    def _extract_visibility(self, landmarks) -> np.ndarray:
        """Extract visibility scores for landmarks"""
        visibility = []
        for landmark in landmarks.landmark:
            visibility.append(landmark.visibility)
        return np.array(visibility)
    
    def _extract_world_landmarks(self, world_landmarks) -> np.ndarray:
        """Extract 3D world landmarks coordinates"""
        coords = []
        for landmark in world_landmarks.landmark:
            coords.append([landmark.x, landmark.y, landmark.z])
        return np.array(coords)
    
    def draw_pose(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Draw pose landmarks on frame
        
        Args:
            frame: Input frame
            pose_data: Pose data from detect_pose
            
        Returns:
            Frame with pose landmarks drawn
        """
        if not pose_data['pose_present']:
            return frame
        
        # Create a copy to avoid modifying original
        annotated_frame = frame.copy()
        
        # Convert landmarks back to MediaPipe format for drawing
        landmarks = pose_data['landmarks']
        if landmarks is not None:
            # Create MediaPipe landmarks object
            mp_landmarks = self.mp_pose.PoseLandmark
            
            # Draw landmarks
            for i, (x, y) in enumerate(landmarks):
                # Convert normalized coordinates to pixel coordinates
                h, w = frame.shape[:2]
                px, py = int(x * w), int(y * h)
                
                # Draw landmark
                cv2.circle(annotated_frame, (px, py), 3, (0, 255, 0), -1)
            
            # Draw connections
            self._draw_connections(annotated_frame, landmarks, frame.shape)
        
        return annotated_frame
    
    def _draw_connections(self, frame: np.ndarray, landmarks: np.ndarray, shape: Tuple):
        """Draw connections between pose landmarks"""
        h, w = shape[:2]
        
        # Define pose connections
        connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            # Body
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (11, 23), (12, 24), (23, 24),
            # Legs
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                # Convert to pixel coordinates
                start_px = (int(start_point[0] * w), int(start_point[1] * h))
                end_px = (int(end_point[0] * w), int(end_point[1] * h))
                
                # Draw line
                cv2.line(frame, start_px, end_px, (0, 255, 255), 2)
    
    def get_body_parts(self, pose_data: Dict) -> Dict[str, np.ndarray]:
        """
        Extract specific body parts from pose data
        
        Args:
            pose_data: Pose data from detect_pose
            
        Returns:
            Dictionary with body parts coordinates
        """
        if not pose_data['pose_present']:
            return {}
        
        landmarks = pose_data['landmarks']
        if landmarks is None:
            return {}
        
        body_parts = {
            'head': landmarks[0:11],  # Nose to mouth
            'torso': landmarks[11:17],  # Shoulders to wrists
            'left_arm': landmarks[[11, 13, 15, 17, 19, 21]],  # Left shoulder to fingers
            'right_arm': landmarks[[12, 14, 16, 18, 20, 22]],  # Right shoulder to fingers
            'hips': landmarks[23:25],  # Hip landmarks
            'left_leg': landmarks[[23, 25, 27, 29, 31]],  # Left hip to foot
            'right_leg': landmarks[[24, 26, 28, 30, 32]]  # Right hip to foot
        }
        
        return body_parts
    
    def calculate_pose_similarity(self, pose1: Dict, pose2: Dict) -> float:
        """
        Calculate similarity between two poses
        
        Args:
            pose1: First pose data
            pose2: Second pose data
            
        Returns:
            Similarity score between 0 and 1
        """
        if not (pose1['pose_present'] and pose2['pose_present']):
            return 0.0
        
        landmarks1 = pose1['landmarks']
        landmarks2 = pose2['landmarks']
        
        if landmarks1 is None or landmarks2 is None:
            return 0.0
        
        # Calculate Euclidean distance between corresponding landmarks
        distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
        
        # Convert to similarity score (inverse of average distance)
        avg_distance = np.mean(distances)
        similarity = 1.0 / (1.0 + avg_distance)
        
        return similarity
    
    def normalize_pose(self, pose_data: Dict) -> Dict:
        """
        Normalize pose relative to body center and scale
        
        Args:
            pose_data: Raw pose data
            
        Returns:
            Normalized pose data
        """
        if not pose_data['pose_present']:
            return pose_data
        
        landmarks = pose_data['landmarks']
        if landmarks is None:
            return pose_data
        
        # Calculate body center (midpoint of shoulders and hips)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        center = np.mean([left_shoulder, right_shoulder, left_hip, right_hip], axis=0)
        
        # Calculate body scale (distance between shoulders)
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_distance == 0:
            shoulder_distance = 1.0
        
        # Normalize landmarks
        normalized_landmarks = (landmarks - center) / shoulder_distance
        
        # Create normalized pose data
        normalized_pose = pose_data.copy()
        normalized_pose['landmarks'] = normalized_landmarks
        normalized_pose['center'] = center
        normalized_pose['scale'] = shoulder_distance
        
        return normalized_pose
    
    def track_pose_sequence(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Track pose through a sequence of frames
        
        Args:
            frames: List of video frames
            
        Returns:
            List of pose data for each frame
        """
        pose_sequence = []
        
        for i, frame in enumerate(frames):
            pose_data = self.detect_pose(frame)
            
            # Add temporal information
            pose_data['frame_index'] = i
            pose_data['timestamp'] = i / 30.0  # Assuming 30 FPS
            
            # Add smoothing if previous pose exists
            if i > 0 and pose_sequence[-1]['pose_present'] and pose_data['pose_present']:
                pose_data = self._smooth_pose(pose_sequence[-1], pose_data)
            
            pose_sequence.append(pose_data)
        
        return pose_sequence
    
    def _smooth_pose(self, prev_pose: Dict, curr_pose: Dict, alpha: float = 0.7) -> Dict:
        """
        Apply temporal smoothing between consecutive poses
        
        Args:
            prev_pose: Previous frame pose data
            curr_pose: Current frame pose data
            alpha: Smoothing factor (0-1)
            
        Returns:
            Smoothed pose data
        """
        if prev_pose['landmarks'] is None or curr_pose['landmarks'] is None:
            return curr_pose
        
        # Apply exponential smoothing
        smoothed_landmarks = (
            alpha * curr_pose['landmarks'] + 
            (1 - alpha) * prev_pose['landmarks']
        )
        
        # Create smoothed pose data
        smoothed_pose = curr_pose.copy()
        smoothed_pose['landmarks'] = smoothed_landmarks
        
        return smoothed_pose
    
    def save_learned_patterns(self, poses: List[Dict], output_path: str) -> None:
        """
        Save learned pose patterns to file
        
        Args:
            poses: List of pose data
            output_path: Path to save the patterns
        """
        try:
            # Extract and normalize poses
            normalized_poses = []
            for pose in poses:
                if pose['pose_present']:
                    normalized_pose = self.normalize_pose(pose)
                    normalized_poses.append(normalized_pose)
            
            # Save to file
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'poses': normalized_poses,
                    'count': len(normalized_poses),
                    'landmarks_mapping': self.pose_landmarks
                }, f)
            
            logger.info(f"Saved {len(normalized_poses)} pose patterns to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving learned patterns: {str(e)}")
    
    def load_learned_patterns(self, input_path: str) -> bool:
        """
        Load learned pose patterns from file
        
        Args:
            input_path: Path to load the patterns from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            
            self.learned_patterns = data['poses']
            logger.info(f"Loaded {len(self.learned_patterns)} pose patterns from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading learned patterns: {str(e)}")
            return False
    
    def find_similar_pose(self, target_pose: Dict, threshold: float = 0.8) -> Optional[Dict]:
        """
        Find similar pose from learned patterns
        
        Args:
            target_pose: Target pose to match
            threshold: Similarity threshold
            
        Returns:
            Most similar pose if found, None otherwise
        """
        if not self.learned_patterns:
            return None
        
        best_similarity = 0.0
        best_pose = None
        
        for learned_pose in self.learned_patterns:
            similarity = self.calculate_pose_similarity(target_pose, learned_pose)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_pose = learned_pose
        
        return best_pose
    
    def estimate_pose_confidence(self, pose_data: Dict) -> float:
        """
        Estimate confidence score for detected pose
        
        Args:
            pose_data: Pose data from detect_pose
            
        Returns:
            Confidence score between 0 and 1
        """
        if not pose_data['pose_present']:
            return 0.0
        
        if pose_data['visibility'] is None:
            return 0.5
        
        # Calculate average visibility of key landmarks
        key_landmarks = [11, 12, 23, 24]  # Shoulders and hips
        key_visibility = pose_data['visibility'][key_landmarks]
        
        return np.mean(key_visibility)
    
    def get_pose_angles(self, pose_data: Dict) -> Dict[str, float]:
        """
        Calculate joint angles from pose data
        
        Args:
            pose_data: Pose data from detect_pose
            
        Returns:
            Dictionary of joint angles in degrees
        """
        if not pose_data['pose_present'] or pose_data['landmarks'] is None:
            return {}
        
        landmarks = pose_data['landmarks']
        angles = {}
        
        # Left arm angle (shoulder-elbow-wrist)
        if len(landmarks) > 15:
            angles['left_arm'] = self._calculate_angle(
                landmarks[11], landmarks[13], landmarks[15]
            )
        
        # Right arm angle (shoulder-elbow-wrist)
        if len(landmarks) > 16:
            angles['right_arm'] = self._calculate_angle(
                landmarks[12], landmarks[14], landmarks[16]
            )
        
        # Left leg angle (hip-knee-ankle)
        if len(landmarks) > 27:
            angles['left_leg'] = self._calculate_angle(
                landmarks[23], landmarks[25], landmarks[27]
            )
        
        # Right leg angle (hip-knee-ankle)
        if len(landmarks) > 28:
            angles['right_leg'] = self._calculate_angle(
                landmarks[24], landmarks[26], landmarks[28]
            )
        
        # Torso angle (shoulder-hip)
        if len(landmarks) > 24:
            shoulder_center = (landmarks[11] + landmarks[12]) / 2
            hip_center = (landmarks[23] + landmarks[24]) / 2
            angles['torso'] = self._calculate_vertical_angle(shoulder_center, hip_center)
        
        return angles
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle between three points
        
        Args:
            p1, p2, p3: Three points as numpy arrays
            
        Returns:
            Angle in degrees
        """
        # Vector from p2 to p1
        v1 = p1 - p2
        # Vector from p2 to p3
        v2 = p3 - p2
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def _calculate_vertical_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate angle from vertical
        
        Args:
            p1, p2: Two points as numpy arrays
            
        Returns:
            Angle from vertical in degrees
        """
        # Vector from p1 to p2
        v = p2 - p1
        
        # Vertical reference vector
        vertical = np.array([0, 1])
        
        # Calculate angle with vertical
        cos_angle = np.dot(v, vertical) / (np.linalg.norm(v) * np.linalg.norm(vertical))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)