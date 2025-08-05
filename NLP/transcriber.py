"""
Audio transcription module using OpenAI Whisper for speech-to-text conversion.
"""

import whisper
import os
from datetime import datetime
import json


class AudioTranscriber:
    def __init__(self, model_size="base"):
        """
        Initialize the transcriber with a Whisper model.
        Model sizes: tiny, base, small, medium, large
        """
        self.model = whisper.load_model(model_size)
        
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file to text with timestamps and confidence scores.
        """
        try:
            result = self.model.transcribe(audio_path, verbose=False)
            
            transcription_data = {
                'timestamp': datetime.now().isoformat(),
                'audio_file': os.path.basename(audio_path),
                'text': result['text'].strip(),
                'language': result['language'],
                'segments': []
            }
            
            # Extract segment-level data for detailed analysis
            for segment in result['segments']:
                transcription_data['segments'].append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'confidence': segment.get('confidence', 0.0)
                })
            
            return transcription_data
            
        except Exception as e:
            print(f"Transcription error for {audio_path}: {str(e)}")
            return None
    
    def batch_transcribe(self, audio_directory):
        """
        Transcribe all audio files in a directory.
        """
        transcriptions = []
        
        for filename in os.listdir(audio_directory):
            if filename.endswith('.wav'):
                audio_path = os.path.join(audio_directory, filename)
                result = self.transcribe_audio(audio_path)
                if result:
                    transcriptions.append(result)
        
        return transcriptions
    
    def save_transcription(self, transcription_data, output_path):
        """
        Save transcription data to JSON file.
        """
        with open(output_path, 'w') as f:
            json.dump(transcription_data, f, indent=2)


if __name__ == "__main__":
    # Test the transcriber
    transcriber = AudioTranscriber()
    
    # Example usage
    audio_file = "../backend/datastorage/example.wav"
    if os.path.exists(audio_file):
        result = transcriber.transcribe_audio(audio_file)
        print("Transcription:", result['text'] if result else "Failed")