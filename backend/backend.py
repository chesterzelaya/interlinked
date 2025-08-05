from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app, origins=['http://127.0.0.1:8000', 'http://localhost:8000'])

# Define the data storage path
DATA_STORAGE_PATH = os.path.join(os.path.dirname(__file__), 'datastorage')

# Ensure DataStorage directory exists
os.makedirs(DATA_STORAGE_PATH, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate filename with readable date and time
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'{timestamp}.wav'
        
        # Save file to DataStorage directory
        file_path = os.path.join(DATA_STORAGE_PATH, filename)
        audio_file.save(file_path)
        
        print(f"Audio file saved: {file_path}")
        
        return jsonify({
            'message': 'Audio uploaded successfully',
            'filename': filename,
            'timestamp': timestamp
        }), 200
        
    except Exception as e:
        print(f"Error uploading audio: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/stats', methods=['GET'])
def get_recording_stats():
    try:
        # Get all wav files in the DataStorage directory
        if not os.path.exists(DATA_STORAGE_PATH):
            return jsonify({}), 200
            
        files = [f for f in os.listdir(DATA_STORAGE_PATH) if f.endswith('.wav')]
        
        # Count recordings per date
        date_counts = {}
        for filename in files:
            try:
                # Extract date from filename (YYYY-MM-DD format)
                date_part = filename.split('_')[0]  # Get date part before underscore
                if date_part in date_counts:
                    date_counts[date_part] += 1
                else:
                    date_counts[date_part] = 1
            except:
                continue  # Skip files that don't match expected format
                
        return jsonify(date_counts), 200
        
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print(f"Data storage path: {DATA_STORAGE_PATH}")
    print("Starting Interlinked backend server...")
    app.run(host='127.0.0.1', port=5000, debug=True)