"""
Main NLP analysis orchestrator for the Interlinked mind monitoring system.
Coordinates all analysis modules and maintains baseline profiles for anomaly detection.
"""

import os
import json
import pickle
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from transcriber import AudioTranscriber
from voice_analysis import VoiceAnalyzer
from sentiment_analysis import SentimentAnalyzer
from embeddings import EmbeddingsAnalyzer
from anomaly_detection import AnomalyDetector


class InterlinkedAnalyzer:
    def __init__(self, data_storage_path=None, analysis_storage_path="analysis_data"):
        """
        Initialize the complete NLP analysis pipeline.
        """
        # Auto-detect the correct data storage path
        if data_storage_path is None:
            # Try different possible paths based on current working directory
            possible_paths = [
                "../backend/datastorage",  # When run from NLP directory
                "backend/datastorage",     # When run from project root
                "./backend/datastorage"    # Alternative project root syntax
            ]
            
            self.data_storage_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.data_storage_path = path
                    break
            
            if self.data_storage_path is None:
                self.data_storage_path = "../backend/datastorage"  # Default fallback
        else:
            self.data_storage_path = data_storage_path
            
        self.analysis_storage_path = analysis_storage_path
        
        # Create analysis storage directory
        os.makedirs(self.analysis_storage_path, exist_ok=True)
        
        # Initialize all analysis modules
        self.transcriber = AudioTranscriber()
        self.voice_analyzer = VoiceAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.embeddings_analyzer = EmbeddingsAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
        # Paths for persistent data
        self.baseline_profile_path = os.path.join(self.analysis_storage_path, "baseline_profile.json")
        self.analysis_history_path = os.path.join(self.analysis_storage_path, "analysis_history.json")
        self.anomaly_reports_path = os.path.join(self.analysis_storage_path, "anomaly_reports.json")
        
        # Load existing data
        self.baseline_profile = self._load_baseline_profile()
        self.analysis_history = self._load_analysis_history()
        self.anomaly_reports = self._load_anomaly_reports()
    
    def analyze_latest_recording(self) -> Dict:
        """
        Analyze the most recent audio recording with complete NLP pipeline.
        """
        # Find the latest audio file
        latest_audio = self._get_latest_audio_file()
        if not latest_audio:
            return {"error": "No audio files found"}
        
        return self.analyze_audio_file(latest_audio)
    
    def analyze_audio_file(self, audio_path: str) -> Dict:
        """
        Run complete analysis pipeline on a specific audio file.
        """
        try:
            print(f"Starting analysis of {os.path.basename(audio_path)}...")
            
            # Step 1: Transcription
            print("  - Transcribing audio...")
            transcription_data = self.transcriber.transcribe_audio(audio_path)
            if not transcription_data:
                return {"error": "Transcription failed"}
            
            # Step 2: Voice Analysis
            print("  - Analyzing voice characteristics...")
            voice_analysis = self.voice_analyzer.analyze_audio(audio_path)
            if not voice_analysis:
                return {"error": "Voice analysis failed"}
            
            # Step 3: Sentiment Analysis
            print("  - Performing sentiment analysis...")
            sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(transcription_data)
            if not sentiment_analysis:
                return {"error": "Sentiment analysis failed"}
            
            # Step 4: Embeddings Analysis
            print("  - Creating semantic embeddings...")
            embeddings_analysis = self.embeddings_analyzer.create_embeddings(transcription_data)
            if not embeddings_analysis:
                return {"error": "Embeddings analysis failed"}
            
            # Combine all analyses
            complete_analysis = {
                'timestamp': datetime.now().isoformat(),
                'audio_file': os.path.basename(audio_path),
                'transcription': transcription_data,
                'voice_analysis': voice_analysis,
                'sentiment_analysis': sentiment_analysis,
                'embeddings_analysis': embeddings_analysis
            }
            
            # Step 5: Anomaly Detection
            print("  - Detecting anomalies...")
            anomaly_report = self.anomaly_detector.detect_anomalies(
                complete_analysis, 
                self.baseline_profile,
                self.analysis_history[-10:] if len(self.analysis_history) >= 10 else None
            )
            
            complete_analysis['anomaly_report'] = anomaly_report
            
            # Step 6: Update baseline and history
            print("  - Updating profiles...")
            self._update_analysis_history(complete_analysis)
            self._update_baseline_profile(complete_analysis)
            
            # Step 7: Store anomaly report if significant
            if anomaly_report.get('anomaly_detected', False):
                self._store_anomaly_report(anomaly_report)
            
            # Step 8: Save analysis
            self._save_complete_analysis(complete_analysis)
            
            print(f"  - Analysis complete! Anomaly detected: {anomaly_report.get('anomaly_detected', False)}")
            
            return self._create_summary_report(complete_analysis)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"  - ERROR: {error_msg}")
            return {"error": error_msg}
    
    def get_current_status(self) -> Dict:
        """
        Get current system status and recent anomaly information.
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'baseline_established': self.baseline_profile is not None,
            'total_recordings_analyzed': len(self.analysis_history),
            'recent_anomalies': len([r for r in self.anomaly_reports if self._is_recent(r['timestamp'])]),
            'last_analysis': None,
            'system_health': 'healthy'
        }
        
        # Get last analysis info
        if self.analysis_history:
            last_analysis = self.analysis_history[-1]
            status['last_analysis'] = {
                'timestamp': last_analysis['timestamp'],
                'audio_file': last_analysis['audio_file'],
                'anomaly_detected': last_analysis.get('anomaly_report', {}).get('anomaly_detected', False),
                'risk_level': last_analysis.get('anomaly_report', {}).get('risk_level', 'low')
            }
        
        # Assess system health based on recent anomalies
        recent_high_risk = len([r for r in self.anomaly_reports 
                              if self._is_recent(r['timestamp']) and r.get('risk_level') == 'high'])
        
        if recent_high_risk > 0:
            status['system_health'] = 'at_risk'
        elif status['recent_anomalies'] > 3:
            status['system_health'] = 'monitoring'
        
        return status
    
    def get_trend_analysis(self, days: int = 7) -> Dict:
        """
        Analyze trends over the specified number of days.
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_analyses = [
            analysis for analysis in self.analysis_history
            if datetime.fromisoformat(analysis['timestamp']) > cutoff_date
        ]
        
        if not recent_analyses:
            return {"error": "No recent analyses found"}
        
        # Extract trend metrics
        sentiment_scores = []
        manipulation_ratios = []
        voice_stress_indicators = []
        semantic_coherence_scores = []
        
        for analysis in recent_analyses:
            # Sentiment trends
            sentiment = analysis.get('sentiment_analysis', {}).get('overall_sentiment', {})
            sentiment_scores.append(sentiment.get('vader_compound', 0))
            
            # Manipulation marker trends
            manip = analysis.get('sentiment_analysis', {}).get('manipulation_markers', {})
            manipulation_ratios.append(manip.get('overall_manipulation_ratio', 0))
            
            # Voice stress trends
            voice = analysis.get('voice_analysis', {})
            jitter = voice.get('jitter', 0)
            shimmer = voice.get('shimmer', 0)
            voice_stress_indicators.append((jitter + shimmer) / 2)
            
            # Semantic coherence trends
            embeddings = analysis.get('embeddings_analysis', {}).get('semantic_coherence', {})
            semantic_coherence_scores.append(embeddings.get('coherence_score', 1.0))
        
        trends = {
            'period_days': days,
            'total_recordings': len(recent_analyses),
            'sentiment_trend': {
                'mean': float(np.mean(sentiment_scores)),
                'std': float(np.std(sentiment_scores)),
                'trend_direction': self._calculate_trend_direction(sentiment_scores)
            },
            'manipulation_trend': {
                'mean': float(np.mean(manipulation_ratios)),
                'std': float(np.std(manipulation_ratios)),
                'trend_direction': self._calculate_trend_direction(manipulation_ratios)
            },
            'voice_stress_trend': {
                'mean': float(np.mean(voice_stress_indicators)),
                'std': float(np.std(voice_stress_indicators)),
                'trend_direction': self._calculate_trend_direction(voice_stress_indicators)
            },
            'semantic_coherence_trend': {
                'mean': float(np.mean(semantic_coherence_scores)),
                'std': float(np.std(semantic_coherence_scores)),
                'trend_direction': self._calculate_trend_direction(semantic_coherence_scores)
            }
        }
        
        # Overall trend assessment
        concerning_trends = []
        if trends['manipulation_trend']['trend_direction'] == 'increasing':
            concerning_trends.append('increasing_manipulation_markers')
        if trends['voice_stress_trend']['trend_direction'] == 'increasing':
            concerning_trends.append('increasing_voice_stress')
        if trends['semantic_coherence_trend']['trend_direction'] == 'decreasing':
            concerning_trends.append('decreasing_semantic_coherence')
        
        trends['concerning_trends'] = concerning_trends
        trends['overall_trend_health'] = 'stable' if not concerning_trends else 'concerning'
        
        return trends
    
    def _get_latest_audio_file(self) -> Optional[str]:
        """
        Find the most recent audio file in the data storage directory.
        """
        if not os.path.exists(self.data_storage_path):
            return None
        
        audio_files = [f for f in os.listdir(self.data_storage_path) if f.endswith('.wav')]
        if not audio_files:
            return None
        
        # Sort by modification time
        audio_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.data_storage_path, f)), reverse=True)
        
        return os.path.join(self.data_storage_path, audio_files[0])
    
    def _load_baseline_profile(self) -> Optional[Dict]:
        """
        Load the baseline profile from disk.
        """
        if os.path.exists(self.baseline_profile_path):
            try:
                with open(self.baseline_profile_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading baseline profile: {e}")
        return None
    
    def _save_baseline_profile(self):
        """
        Save the baseline profile to disk.
        """
        try:
            with open(self.baseline_profile_path, 'w') as f:
                json.dump(self.baseline_profile, f, indent=2)
        except Exception as e:
            print(f"Error saving baseline profile: {e}")
    
    def _load_analysis_history(self) -> List[Dict]:
        """
        Load analysis history from disk.
        """
        if os.path.exists(self.analysis_history_path):
            try:
                with open(self.analysis_history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading analysis history: {e}")
        return []
    
    def _save_analysis_history(self):
        """
        Save analysis history to disk.
        """
        try:
            # Keep only the last 100 analyses to prevent file bloat
            recent_history = self.analysis_history[-100:] if len(self.analysis_history) > 100 else self.analysis_history
            
            with open(self.analysis_history_path, 'w') as f:
                json.dump(recent_history, f, indent=2)
        except Exception as e:
            print(f"Error saving analysis history: {e}")
    
    def _load_anomaly_reports(self) -> List[Dict]:
        """
        Load anomaly reports from disk.
        """
        if os.path.exists(self.anomaly_reports_path):
            try:
                with open(self.anomaly_reports_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading anomaly reports: {e}")
        return []
    
    def _save_anomaly_reports(self):
        """
        Save anomaly reports to disk.
        """
        try:
            with open(self.anomaly_reports_path, 'w') as f:
                json.dump(self.anomaly_reports, f, indent=2)
        except Exception as e:
            print(f"Error saving anomaly reports: {e}")
    
    def _update_baseline_profile(self, analysis: Dict):
        """
        Update the baseline profile with new analysis data.
        """
        if not self.baseline_profile:
            # Create initial baseline from first few analyses
            if len(self.analysis_history) >= 3:
                self.baseline_profile = self._create_baseline_from_history()
        else:
            # Update existing baseline with exponential moving average
            self._update_baseline_with_analysis(analysis)
        
        self._save_baseline_profile()
    
    def _create_baseline_from_history(self) -> Dict:
        """
        Create initial baseline profile from first few analyses.
        """
        if len(self.analysis_history) < 3:
            return None
        
        baseline_analyses = self.analysis_history[:5]  # Use first 5 recordings
        
        # Aggregate voice features
        voice_features = {}
        voice_feature_names = ['pitch_mean', 'pitch_std', 'speaking_rate', 'energy_mean', 'jitter', 'shimmer']
        
        for feature in voice_feature_names:
            values = [analysis.get('voice_analysis', {}).get(feature, 0) for analysis in baseline_analyses]
            voice_features[feature] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        # Aggregate sentiment features
        sentiment_features = {
            'overall_sentiment': {},
            'manipulation_markers': {},
            'emotional_volatility': {}
        }
        
        # Calculate averages for sentiment metrics
        vader_compounds = [analysis.get('sentiment_analysis', {}).get('overall_sentiment', {}).get('vader_compound', 0) 
                          for analysis in baseline_analyses]
        sentiment_features['overall_sentiment']['vader_compound'] = float(np.mean(vader_compounds))
        
        manip_ratios = [analysis.get('sentiment_analysis', {}).get('manipulation_markers', {}).get('overall_manipulation_ratio', 0)
                       for analysis in baseline_analyses]
        sentiment_features['manipulation_markers']['overall_manipulation_ratio'] = float(np.mean(manip_ratios))
        
        # Aggregate embeddings features
        embeddings_list = [analysis.get('embeddings_analysis', {}) for analysis in baseline_analyses]
        embeddings_profile = self.embeddings_analyzer.create_baseline_profile(embeddings_list)
        
        baseline_profile = {
            'created_timestamp': datetime.now().isoformat(),
            'version': 1.0,
            'voice_profile': voice_features,
            'sentiment_profile': sentiment_features,
            'embeddings_profile': embeddings_profile,
            'historical_statistics': voice_features,  # For anomaly detection
            'baseline_recordings_count': len(baseline_analyses)
        }
        
        return baseline_profile
    
    def _update_baseline_with_analysis(self, analysis: Dict):
        """
        Update existing baseline profile with exponential moving average.
        """
        alpha = 0.1  # Learning rate for exponential moving average
        
        # Update voice profile
        voice_data = analysis.get('voice_analysis', {})
        voice_profile = self.baseline_profile.get('voice_profile', {})
        
        for feature, current_value in voice_data.items():
            if isinstance(current_value, (int, float)) and feature in voice_profile:
                old_mean = voice_profile[feature].get('mean', current_value)
                new_mean = (1 - alpha) * old_mean + alpha * current_value
                voice_profile[feature]['mean'] = float(new_mean)
        
        # Update sentiment profile
        sentiment_data = analysis.get('sentiment_analysis', {})
        sentiment_profile = self.baseline_profile.get('sentiment_profile', {})
        
        # Update overall sentiment
        current_vader = sentiment_data.get('overall_sentiment', {}).get('vader_compound', 0)
        if 'overall_sentiment' in sentiment_profile:
            old_vader = sentiment_profile['overall_sentiment'].get('vader_compound', current_vader)
            new_vader = (1 - alpha) * old_vader + alpha * current_vader
            sentiment_profile['overall_sentiment']['vader_compound'] = float(new_vader)
    
    def _update_analysis_history(self, analysis: Dict):
        """
        Add new analysis to history.
        """
        self.analysis_history.append(analysis)
        self._save_analysis_history()
    
    def _store_anomaly_report(self, anomaly_report: Dict):
        """
        Store significant anomaly reports.
        """
        self.anomaly_reports.append(anomaly_report)
        self._save_anomaly_reports()
    
    def _save_complete_analysis(self, analysis: Dict):
        """
        Save complete analysis to individual file.
        """
        filename = f"analysis_{analysis['audio_file'].replace('.wav', '')}.json"
        filepath = os.path.join(self.analysis_storage_path, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2)
        except Exception as e:
            print(f"Error saving complete analysis: {e}")
    
    def _create_summary_report(self, analysis: Dict) -> Dict:
        """
        Create a concise summary report for the user.
        """
        anomaly_report = analysis.get('anomaly_report', {})
        
        summary = {
            'timestamp': analysis['timestamp'],
            'audio_file': analysis['audio_file'],
            'status': 'analyzed',
            'text_preview': analysis['transcription']['text'][:100] + "..." if len(analysis['transcription']['text']) > 100 else analysis['transcription']['text'],
            'anomaly_detected': anomaly_report.get('anomaly_detected', False),
            'risk_level': anomaly_report.get('risk_level', 'low'),
            'confidence_score': anomaly_report.get('confidence_score', 0),
            'anomaly_types': anomaly_report.get('anomaly_types', []),
            'recommendations': anomaly_report.get('recommendations', [])
        }
        
        # Add key metrics
        voice_analysis = analysis.get('voice_analysis', {})
        sentiment_analysis = analysis.get('sentiment_analysis', {})
        
        # Safely extract nested values
        def safe_get_nested(data, *keys, default=0):
            """Safely get nested dictionary values"""
            result = data
            for key in keys:
                if isinstance(result, dict):
                    result = result.get(key, {})
                else:
                    return default
            return result if not isinstance(result, dict) else default
        
        # Handle emotional volatility which might be a plain number or dict
        emotional_volatility_data = sentiment_analysis.get('emotional_volatility', 0)
        if isinstance(emotional_volatility_data, dict):
            emotional_volatility_value = emotional_volatility_data.get('compound_volatility', 0)
        else:
            emotional_volatility_value = emotional_volatility_data  # Use the plain number
        
        summary['key_metrics'] = {
            'speaking_rate': voice_analysis.get('speaking_rate', 0),
            'sentiment_score': safe_get_nested(sentiment_analysis, 'overall_sentiment', 'vader_compound'),
            'manipulation_ratio': safe_get_nested(sentiment_analysis, 'manipulation_markers', 'overall_manipulation_ratio'),
            'emotional_volatility': emotional_volatility_value
        }
        
        return summary
    
    def _is_recent(self, timestamp: str, hours: int = 24) -> bool:
        """
        Check if timestamp is within the last N hours.
        """
        try:
            ts = datetime.fromisoformat(timestamp)
            return datetime.now() - ts < timedelta(hours=hours)
        except:
            return False
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """
        Calculate trend direction using linear regression slope.
        """
        if len(values) < 2:
            return 'stable'
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'


if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = InterlinkedAnalyzer()
    
    # Analyze the latest recording
    result = analyzer.analyze_latest_recording()
    
    if 'error' in result:
        print(f"Analysis failed: {result['error']}")
    else:
        print("\n=== INTERLINKED ANALYSIS REPORT ===")
        print(f"File: {result['audio_file']}")
        print(f"Status: {result['status']}")
        print(f"\nText Preview: {result['text_preview']}")
        print(f"\nAnomaly Detected: {result['anomaly_detected']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        
        if result['anomaly_types']:
            print(f"\nAnomaly Types: {', '.join(result['anomaly_types'])}")
        
        if result['recommendations']:
            print("\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  - {rec}")
        
        print(f"\nKey Metrics:")
        metrics = result['key_metrics']
        print(f"  Speaking Rate: {metrics['speaking_rate']:.2f}")
        print(f"  Sentiment Score: {metrics['sentiment_score']:.2f}")
        print(f"  Manipulation Ratio: {metrics['manipulation_ratio']:.3f}")
        print(f"  Emotional Volatility: {metrics['emotional_volatility']:.3f}")
    
    # Show system status
    status = analyzer.get_current_status()
    print(f"\n=== SYSTEM STATUS ===")
    print(f"Health: {status['system_health']}")
    print(f"Total Recordings: {status['total_recordings_analyzed']}")
    print(f"Recent Anomalies: {status['recent_anomalies']}")
    print(f"Baseline Established: {status['baseline_established']}")