import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Directory configurations
DATA_DIR = BASE_DIR / "data"
INPUT_VIDEOS_DIR = DATA_DIR / "input_videos"
OUTPUT_VIDEOS_DIR = DATA_DIR / "output_videos"
FACES_DIR = DATA_DIR / "faces"
CLOTHING_DIR = DATA_DIR / "clothing"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
for dir_path in [DATA_DIR, INPUT_VIDEOS_DIR, OUTPUT_VIDEOS_DIR, 
                 FACES_DIR, CLOTHING_DIR, BACKGROUNDS_DIR, 
                 MODELS_DIR, WEIGHTS_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
POSE_MODEL_CONFIG = {
    'model_complexity': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'static_image_mode': False,
    'smooth_landmarks': True
}

FACE_MODEL_CONFIG = {
    'face_detection_confidence': 0.6,
    'face_recognition_tolerance': 0.6,
    'face_alignment': True,
    'face_enhancement': True
}

SEGMENTATION_CONFIG = {
    'model_name': 'deeplabv3_resnet50',
    'device': 'cuda' if os.getenv('CUDA_AVAILABLE', 'false').lower() == 'true' else 'cpu',
    'confidence_threshold': 0.7
}

# Video processing settings
VIDEO_CONFIG = {
    'output_fps': 30,
    'output_quality': 'high',
    'max_resolution': (1920, 1080),
    'compression': 'h264',
    'audio_enabled': True
}

# Background removal settings
BACKGROUND_CONFIG = {
    'model': 'u2net',
    'alpha_matting': True,
    'alpha_matting_foreground_threshold': 270,
    'alpha_matting_background_threshold': 10,
    'alpha_matting_erode_size': 10
}

# Clothing manipulation settings
CLOTHING_CONFIG = {
    'body_parts': ['torso', 'arms', 'legs'],
    'clothing_types': ['shirt', 'pants', 'dress', 'jacket', 'accessories'],
    'fitting_algorithm': 'deformation_based',
    'texture_quality': 'high'
}

# Text processing settings
TEXT_CONFIG = {
    'model': 'gpt-3.5-turbo',
    'max_tokens': 150,
    'temperature': 0.7,
    'language': 'en'
}

# Web interface settings
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'max_file_size': 500 * 1024 * 1024,  # 500MB
    'allowed_extensions': ['mp4', 'avi', 'mov', 'wmv', 'flv']
}

# GPU settings
GPU_CONFIG = {
    'use_gpu': True,
    'gpu_memory_fraction': 0.8,
    'mixed_precision': True,
    'batch_size': 4
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': BASE_DIR / 'logs' / 'app.log'
}

# Create logs directory
(BASE_DIR / 'logs').mkdir(exist_ok=True)

# API keys (set these in environment variables)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')

# Model URLs for download
MODEL_URLS = {
    'pose_model': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
    'face_model': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
    'segmentation_model': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth'
}