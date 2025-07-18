#!/usr/bin/env python3
"""
Video Generation App for Clothing Business
Main application entry point
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils.video_processor import VideoProcessor
from utils.pose_estimator import PoseEstimator
from utils.face_swapper import FaceSwapper
from utils.clothing_changer import ClothingChanger
from utils.background_remover import BackgroundRemover
from models.pose_model import PoseModel
from models.face_model import FaceModel
from models.segmentation_model import SegmentationModel

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VideoGenerator:
    """Main video generation class"""
    
    def __init__(self):
        """Initialize the video generator with all required models"""
        logger.info("Initializing Video Generator...")
        
        # Initialize models
        self.pose_model = PoseModel()
        self.face_model = FaceModel()
        self.segmentation_model = SegmentationModel()
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.pose_estimator = PoseEstimator(self.pose_model)
        self.face_swapper = FaceSwapper(self.face_model)
        self.clothing_changer = ClothingChanger(self.segmentation_model)
        self.background_remover = BackgroundRemover()
        
        logger.info("Video Generator initialized successfully!")
    
    def process_video(self, 
                     input_path: str,
                     face_image: Optional[str] = None,
                     clothing_style: Optional[str] = None,
                     background: Optional[str] = None,
                     text_prompt: Optional[str] = None,
                     output_path: Optional[str] = None) -> str:
        """
        Process a video with the specified modifications
        
        Args:
            input_path: Path to input video
            face_image: Path to face image for swapping
            clothing_style: Style of clothing to apply
            background: Path to background image
            text_prompt: Text description for modifications
            output_path: Path for output video
            
        Returns:
            Path to generated video
        """
        try:
            logger.info(f"Processing video: {input_path}")
            
            # Validate input
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input video not found: {input_path}")
            
            # Generate output path if not provided
            if output_path is None:
                output_path = str(OUTPUT_VIDEOS_DIR / f"generated_{Path(input_path).stem}.mp4")
            
            # Load video
            video_data = self.video_processor.load_video(input_path)
            frames = video_data['frames']
            fps = video_data['fps']
            
            logger.info(f"Loaded video with {len(frames)} frames at {fps} FPS")
            
            # Process text prompt if provided
            if text_prompt:
                modifications = self._parse_text_prompt(text_prompt)
                if 'face' in modifications:
                    face_image = modifications['face']
                if 'clothing' in modifications:
                    clothing_style = modifications['clothing']
                if 'background' in modifications:
                    background = modifications['background']
            
            # Process each frame
            processed_frames = []
            
            for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
                processed_frame = self._process_frame(
                    frame, 
                    face_image=face_image,
                    clothing_style=clothing_style,
                    background=background
                )
                processed_frames.append(processed_frame)
            
            # Save processed video
            self.video_processor.save_video(
                processed_frames, 
                output_path, 
                fps=fps
            )
            
            logger.info(f"Video saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
    
    def _process_frame(self, 
                      frame: np.ndarray,
                      face_image: Optional[str] = None,
                      clothing_style: Optional[str] = None,
                      background: Optional[str] = None) -> np.ndarray:
        """Process a single frame with modifications"""
        
        result_frame = frame.copy()
        
        try:
            # Pose estimation
            pose_data = self.pose_estimator.detect_pose(frame)
            
            # Face swapping
            if face_image and os.path.exists(face_image):
                result_frame = self.face_swapper.swap_face(
                    result_frame, 
                    face_image, 
                    pose_data
                )
            
            # Clothing change
            if clothing_style:
                result_frame = self.clothing_changer.change_clothing(
                    result_frame, 
                    clothing_style, 
                    pose_data
                )
            
            # Background replacement
            if background and os.path.exists(background):
                result_frame = self.background_remover.replace_background(
                    result_frame, 
                    background
                )
            
            return result_frame
            
        except Exception as e:
            logger.warning(f"Error processing frame: {str(e)}")
            return frame
    
    def _parse_text_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse text prompt to extract modifications"""
        modifications = {}
        
        # Simple keyword-based parsing (can be enhanced with NLP)
        prompt_lower = prompt.lower()
        
        # Face-related keywords
        if any(word in prompt_lower for word in ['face', 'head', 'person']):
            modifications['face'] = 'auto'
        
        # Clothing-related keywords
        clothing_keywords = {
            'formal': 'formal_suit',
            'casual': 'casual_wear',
            'dress': 'dress',
            'suit': 'business_suit',
            'shirt': 'shirt',
            'jacket': 'jacket'
        }
        
        for keyword, style in clothing_keywords.items():
            if keyword in prompt_lower:
                modifications['clothing'] = style
                break
        
        # Background-related keywords
        background_keywords = {
            'beach': 'beach.jpg',
            'studio': 'studio.jpg',
            'office': 'office.jpg',
            'outdoor': 'outdoor.jpg'
        }
        
        for keyword, bg in background_keywords.items():
            if keyword in prompt_lower:
                modifications['background'] = str(BACKGROUNDS_DIR / bg)
                break
        
        return modifications
    
    def learn_from_video(self, input_path: str, output_model_path: str) -> None:
        """Learn pose and movement patterns from input video"""
        logger.info(f"Learning from video: {input_path}")
        
        # Load video
        video_data = self.video_processor.load_video(input_path)
        frames = video_data['frames']
        
        # Extract poses from all frames
        poses = []
        for frame in tqdm(frames, desc="Extracting poses"):
            pose_data = self.pose_estimator.detect_pose(frame)
            poses.append(pose_data)
        
        # Save learned patterns
        self.pose_estimator.save_learned_patterns(poses, output_model_path)
        logger.info(f"Learned patterns saved to: {output_model_path}")

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Video Generation App')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--face', '-f', help='Face image path')
    parser.add_argument('--clothing', '-c', help='Clothing style')
    parser.add_argument('--background', '-b', help='Background image path')
    parser.add_argument('--prompt', '-p', help='Text prompt for modifications')
    parser.add_argument('--learn', '-l', action='store_true', help='Learn from input video')
    parser.add_argument('--model-output', '-m', help='Output path for learned model')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = VideoGenerator()
    
    if args.learn:
        # Learning mode
        model_output = args.model_output or str(MODELS_DIR / 'learned_poses.pkl')
        generator.learn_from_video(args.input, model_output)
    else:
        # Generation mode
        output_path = generator.process_video(
            input_path=args.input,
            face_image=args.face,
            clothing_style=args.clothing,
            background=args.background,
            text_prompt=args.prompt,
            output_path=args.output
        )
        print(f"Generated video saved to: {output_path}")

if __name__ == "__main__":
    main()