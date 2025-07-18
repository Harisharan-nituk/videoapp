#!/usr/bin/env python3
"""
Flask Web Application for Video Generation
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from main import VideoGenerator

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format']
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = WEB_CONFIG['max_file_size']
app.config['UPLOAD_FOLDER'] = TEMP_DIR
app.secret_key = 'your-secret-key-here'

# Initialize video generator
video_generator = VideoGenerator()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in WEB_CONFIG['allowed_extensions']

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract video info
            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            return jsonify({
                'success': True,
                'filename': filename,
                'video_info': {
                    'duration': duration,
                    'fps': fps,
                    'frame_count': frame_count,
                    'resolution': [width, height]
                }
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_face', methods=['POST'])
def upload_face():
    """Handle face image upload"""
    try:
        if 'face' not in request.files:
            return jsonify({'error': 'No face image provided'}), 400
        
        file = request.files['face']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"face_{timestamp}_{filename}"
            
            filepath = os.path.join(FACES_DIR, filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': str(filepath)
            })
        
        return jsonify({'error': 'Invalid file'}), 400
        
    except Exception as e:
        logger.error(f"Face upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_background', methods=['POST'])
def upload_background():
    """Handle background image upload"""
    try:
        if 'background' not in request.files:
            return jsonify({'error': 'No background image provided'}), 400
        
        file = request.files['background']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bg_{timestamp}_{filename}"
            
            filepath = os.path.join(BACKGROUNDS_DIR, filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': str(filepath)
            })
        
        return jsonify({'error': 'Invalid file'}), 400
        
    except Exception as e:
        logger.error(f"Background upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_video():
    """Process video with AI modifications"""
    try:
        data = request.get_json()
        
        # Get parameters
        input_filename = data.get('input_filename')
        face_filename = data.get('face_filename')
        background_filename = data.get('background_filename')
        clothing_style = data.get('clothing_style')
        text_prompt = data.get('text_prompt')
        
        if not input_filename:
            return jsonify({'error': 'No input video specified'}), 400
        
        # Build file paths
        input_path = os.path.join(TEMP_DIR, input_filename)
        face_path = os.path.join(FACES_DIR, face_filename) if face_filename else None
        background_path = os.path.join(BACKGROUNDS_DIR, background_filename) if background_filename else None
        
        # Generate output path
        output_filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(OUTPUT_VIDEOS_DIR, output_filename)
        
        # Process video
        result_path = video_generator.process_video(
            input_path=input_path,
            face_image=face_path,
            clothing_style=clothing_style,
            background=background_path,
            text_prompt=text_prompt,
            output_path=output_path
        )
        
        return jsonify({
            'success': True,
            'output_filename': output_filename,
            'output_path': result_path,
            'message': 'Video processed successfully'
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/learn', methods=['POST'])
def learn_from_video():
    """Learn pose patterns from video"""
    try:
        data = request.get_json()
        input_filename = data.get('input_filename')
        
        if not input_filename:
            return jsonify({'error': 'No input video specified'}), 400
        
        input_path = os.path.join(TEMP_DIR, input_filename)
        model_output = os.path.join(MODELS_DIR, f"learned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        
        # Learn from video
        video_generator.learn_from_video(input_path, model_output)
        
        return jsonify({
            'success': True,
            'model_path': model_output,
            'message': 'Learning completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Learning error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated video"""
    try:
        file_path = os.path.join(OUTPUT_VIDEOS_DIR, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get application status"""
    return jsonify({
        'status': 'running',
        'models_loaded': True,
        'gpu_available': GPU_CONFIG['use_gpu'],
        'supported_formats': WEB_CONFIG['allowed_extensions']
    })

@app.route('/api/clothing_styles')
def get_clothing_styles():
    """Get available clothing styles"""
    styles = {
        'formal': {
            'name': 'Formal Suit',
            'description': 'Business formal attire',
            'category': 'formal'
        },
        'casual': {
            'name': 'Casual Wear',
            'description': 'Relaxed casual clothing',
            'category': 'casual'
        },
        'dress': {
            'name': 'Dress',
            'description': 'Elegant dress',
            'category': 'formal'
        },
        'sports': {
            'name': 'Sports Wear',
            'description': 'Athletic clothing',
            'category': 'sports'
        },
        'winter': {
            'name': 'Winter Wear',
            'description': 'Warm winter clothing',
            'category': 'seasonal'
        }
    }
    
    return jsonify(styles)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(BACKGROUNDS_DIR, exist_ok=True)
    
    # Run the application
    app.run(
        host=WEB_CONFIG['host'],
        port=WEB_CONFIG['port'],
        debug=WEB_CONFIG['debug'],
        threaded=True
    )