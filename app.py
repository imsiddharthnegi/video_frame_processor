import os
import logging
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import uuid
import json
from video_processor import VideoProcessor
from vector_db import VectorDB

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Enable CORS
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Initialize video processor and vector database
video_processor = VideoProcessor()
vector_db = VectorDB()

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page for video upload."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video file upload and processing."""
    try:
        if 'video' not in request.files:
            flash('No video file provided', 'error')
            return redirect(url_for('index'))
        
        file = request.files['video']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload MP4, AVI, MOV, or MKV files.', 'error')
            return redirect(url_for('index'))
        
        # Get frame interval from form
        try:
            frame_interval = float(request.form.get('frame_interval', 1.0))
            if frame_interval <= 0:
                frame_interval = 1.0
        except (ValueError, TypeError):
            frame_interval = 1.0
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_extension}"
        
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process video
        app.logger.info(f"Processing video: {unique_filename}")
        frames_info = video_processor.extract_frames(
            filepath, 
            app.config['FRAMES_FOLDER'], 
            file_id, 
            frame_interval
        )
        
        if not frames_info:
            flash('Failed to extract frames from video', 'error')
            return redirect(url_for('index'))
        
        # Compute feature vectors
        app.logger.info(f"Computing feature vectors for {len(frames_info)} frames")
        for frame_info in frames_info:
            feature_vector = video_processor.compute_feature_vector(frame_info['path'])
            if feature_vector is not None:
                vector_db.add_vector(
                    frame_info['id'],
                    feature_vector,
                    {
                        'video_id': file_id,
                        'frame_path': frame_info['path'],
                        'timestamp': frame_info['timestamp'],
                        'frame_number': frame_info['frame_number']
                    }
                )
        
        flash(f'Successfully processed video! Extracted {len(frames_info)} frames.', 'success')
        return redirect(url_for('results', video_id=file_id))
        
    except Exception as e:
        app.logger.error(f"Error processing video: {str(e)}")
        flash(f'Error processing video: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/results/<video_id>')
def results(video_id):
    """Display results for a processed video."""
    try:
        # Get all frames for this video
        video_frames = vector_db.get_frames_by_video(video_id)
        if not video_frames:
            flash('Video not found or no frames extracted', 'error')
            return redirect(url_for('index'))
        
        return render_template('results.html', 
                             video_id=video_id, 
                             frames=video_frames)
    except Exception as e:
        app.logger.error(f"Error displaying results: {str(e)}")
        flash(f'Error displaying results: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/upload', methods=['POST'])
def api_upload_video():
    """API endpoint for video upload."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get frame interval
        frame_interval = float(request.form.get('frame_interval', 1.0))
        if frame_interval <= 0:
            frame_interval = 1.0
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_extension}"
        
        # Save and process
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        frames_info = video_processor.extract_frames(
            filepath, 
            app.config['FRAMES_FOLDER'], 
            file_id, 
            frame_interval
        )
        
        if not frames_info:
            return jsonify({'error': 'Failed to extract frames'}), 500
        
        # Compute and store feature vectors
        processed_frames = []
        for frame_info in frames_info:
            feature_vector = video_processor.compute_feature_vector(frame_info['path'])
            if feature_vector is not None:
                vector_db.add_vector(
                    frame_info['id'],
                    feature_vector,
                    {
                        'video_id': file_id,
                        'frame_path': frame_info['path'],
                        'timestamp': frame_info['timestamp'],
                        'frame_number': frame_info['frame_number']
                    }
                )
                processed_frames.append({
                    'id': frame_info['id'],
                    'timestamp': frame_info['timestamp'],
                    'frame_number': frame_info['frame_number']
                })
        
        return jsonify({
            'success': True,
            'video_id': file_id,
            'frames_count': len(processed_frames),
            'frames': processed_frames
        })
        
    except Exception as e:
        app.logger.error(f"API upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def api_search_similar():
    """API endpoint for similarity search."""
    try:
        data = request.get_json()
        if not data or 'frame_id' not in data:
            return jsonify({'error': 'Frame ID required'}), 400
        
        frame_id = data['frame_id']
        top_k = data.get('top_k', 5)
        
        # Get the feature vector for the query frame
        query_vector = vector_db.get_vector(frame_id)
        if query_vector is None:
            return jsonify({'error': 'Frame not found'}), 404
        
        # Search for similar frames
        similar_frames = vector_db.search_similar(query_vector, top_k + 1)  # +1 to exclude self
        
        # Remove the query frame from results
        similar_frames = [f for f in similar_frames if f['id'] != frame_id][:top_k]
        
        return jsonify({
            'query_frame_id': frame_id,
            'similar_frames': similar_frames
        })
        
    except Exception as e:
        app.logger.error(f"API search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/frame/<frame_id>')
def get_frame(frame_id):
    """Serve frame image by ID."""
    try:
        frame_data = vector_db.get_frame_metadata(frame_id)
        if not frame_data or 'frame_path' not in frame_data:
            return jsonify({'error': 'Frame not found'}), 404
        
        frame_path = frame_data['frame_path']
        if not os.path.exists(frame_path):
            return jsonify({'error': 'Frame file not found'}), 404
        
        return send_file(frame_path, mimetype='image/jpeg')
        
    except Exception as e:
        app.logger.error(f"Error serving frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/frames/<video_id>')
def api_get_frames(video_id):
    """API endpoint to get all frames for a video."""
    try:
        frames = vector_db.get_frames_by_video(video_id)
        return jsonify({
            'video_id': video_id,
            'frames': frames
        })
    except Exception as e:
        app.logger.error(f"API get frames error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/search_similar/<frame_id>')
def search_similar(frame_id):
    """Web interface for similarity search."""
    try:
        query_vector = vector_db.get_vector(frame_id)
        if query_vector is None:
            flash('Frame not found', 'error')
            return redirect(url_for('index'))
        
        similar_frames = vector_db.search_similar(query_vector, 6)  # Get 6 to exclude self
        similar_frames = [f for f in similar_frames if f['id'] != frame_id][:5]
        
        query_frame = vector_db.get_frame_metadata(frame_id)
        
        return render_template('results.html', 
                             query_frame=query_frame,
                             similar_frames=similar_frames,
                             is_similarity_search=True)
        
    except Exception as e:
        app.logger.error(f"Error in similarity search: {str(e)}")
        flash(f'Error in similarity search: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 100MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
