"""
Sentiment analysis module for detecting emotional states and sentiment patterns
in transcribed speech that might indicate external influence or manipulation.
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from collections import Counter
import numpy as np
from datetime import datetime
import json


class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.sia = SentimentIntensityAnalyzer()
        
        # Psychological markers that might indicate manipulation
        self.manipulation_keywords = {
            'uncertainty': ['maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain', 'unsure', 'doubt'],
            'absolute_thinking': ['always', 'never', 'everyone', 'nobody', 'everything', 'nothing', 'all', 'none'],
            'emotional_appeals': ['afraid', 'scared', 'angry', 'outraged', 'disgusted', 'threatened'],
            'social_pressure': ['everyone thinks', 'people say', 'they want', 'should', 'must', 'have to'],
            'urgency': ['now', 'immediately', 'quickly', 'urgent', 'emergency', 'crisis', 'deadline']
        }
    
    def analyze_sentiment(self, transcription_data):
        """
        Comprehensive sentiment analysis of transcribed text.
        """
        try:
            print(f"Starting sentiment analysis...")
            text = transcription_data.get('text', '')
            segments = transcription_data.get('segments', [])
            
            print(f"Text length: {len(text)}")
            print(f"Number of segments: {len(segments)}")
            
            if not text.strip():
                print("No text to analyze")
                return None
            
            # Overall sentiment analysis
            print("Calculating VADER scores...")
            vader_scores = self.sia.polarity_scores(text)
            print("Calculating TextBlob sentiment...")
            textblob_sentiment = TextBlob(text).sentiment
            print("Basic sentiment analysis complete.")
            
            # Segment-level sentiment analysis
            segment_sentiments = []
            for segment in segments:
                seg_text = segment.get('text', '')
                if seg_text.strip():
                    seg_vader = self.sia.polarity_scores(seg_text)
                    seg_textblob = TextBlob(seg_text).sentiment
                    
                    segment_sentiments.append({
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'text': seg_text,
                        'vader_compound': seg_vader['compound'],
                        'vader_pos': seg_vader['pos'],
                        'vader_neu': seg_vader['neu'],
                        'vader_neg': seg_vader['neg'],
                        'textblob_polarity': seg_textblob.polarity,
                        'textblob_subjectivity': seg_textblob.subjectivity
                    })
            
            # Psychological manipulation markers
            manipulation_scores = self._analyze_manipulation_markers(text)
            
            # Linguistic complexity analysis
            complexity_metrics = self._analyze_linguistic_complexity(text)
            
            # Emotional volatility (sentiment changes across segments)
            volatility = self._calculate_emotional_volatility(segment_sentiments)
            
            sentiment_analysis = {
                'timestamp': datetime.now().isoformat(),
                'audio_file': transcription_data.get('audio_file', ''),
                'overall_sentiment': {
                    'vader_compound': vader_scores['compound'],
                    'vader_positive': vader_scores['pos'],
                    'vader_neutral': vader_scores['neu'],
                    'vader_negative': vader_scores['neg'],
                    'textblob_polarity': textblob_sentiment.polarity,
                    'textblob_subjectivity': textblob_sentiment.subjectivity
                },
                'segment_sentiments': segment_sentiments,
                'emotional_volatility': volatility,
                'manipulation_markers': manipulation_scores,
                'linguistic_complexity': complexity_metrics,
                'word_count': len(text.split()),
                'unique_words': len(set(text.lower().split())),
                'lexical_diversity': self._calculate_lexical_diversity(text)
            }
            
            return sentiment_analysis
            
        except Exception as e:
            print(f"Sentiment analysis error: {str(e)}")
            return None
    
    def _analyze_manipulation_markers(self, text):
        """
        Detect linguistic patterns that might indicate external influence.
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = len(words)
        
        manipulation_scores = {}
        
        for category, keywords in self.manipulation_keywords.items():
            matches = sum(1 for word in words if word in keywords)
            manipulation_scores[f'{category}_count'] = matches
            manipulation_scores[f'{category}_ratio'] = matches / word_count if word_count > 0 else 0
        
        # Overall manipulation score
        total_manipulation_words = sum(manipulation_scores[key] for key in manipulation_scores if key.endswith('_count'))
        manipulation_scores['overall_manipulation_ratio'] = total_manipulation_words / word_count if word_count > 0 else 0
        
        return manipulation_scores
    
    def _analyze_linguistic_complexity(self, text):
        """
        Analyze linguistic complexity which might change under external influence.
        """
        sentences = nltk.sent_tokenize(text)
        words = re.findall(r'\b\w+\b', text)
        
        if not sentences or not words:
            return {
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'sentence_count': 0,
                'syllable_complexity': 0
            }
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Syllable complexity (approximation)
        syllable_complexity = self._estimate_syllable_complexity(words)
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'sentence_count': len(sentences),
            'syllable_complexity': syllable_complexity
        }
    
    def _estimate_syllable_complexity(self, words):
        """
        Rough syllable count estimation for complexity analysis.
        """
        vowels = 'aeiouy'
        total_syllables = 0
        
        for word in words:
            word = word.lower().strip()
            syllables = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = is_vowel
            
            # Handle silent e
            if word.endswith('e') and syllables > 1:
                syllables -= 1
            
            # Ensure at least one syllable
            if syllables == 0:
                syllables = 1
            
            total_syllables += syllables
        
        return total_syllables / len(words) if words else 0
    
    def _calculate_emotional_volatility(self, segment_sentiments):
        """
        Calculate how much sentiment varies across the speech segments.
        """
        if len(segment_sentiments) < 2:
            return 0
        
        compound_scores = [seg['vader_compound'] for seg in segment_sentiments]
        polarity_scores = [seg['textblob_polarity'] for seg in segment_sentiments]
        
        # Calculate standard deviation of sentiment scores
        compound_volatility = np.std(compound_scores) if compound_scores else 0
        polarity_volatility = np.std(polarity_scores) if polarity_scores else 0
        
        return {
            'compound_volatility': float(compound_volatility),
            'polarity_volatility': float(polarity_volatility),
            'sentiment_range': float(max(compound_scores) - min(compound_scores)) if compound_scores else 0
        }
    
    def _calculate_lexical_diversity(self, text):
        """
        Calculate lexical diversity (Type-Token Ratio).
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words
    
    def compare_baseline(self, current_analysis, baseline_analysis):
        """
        Compare current sentiment analysis against baseline to detect changes.
        """
        if not baseline_analysis:
            return {}
        
        comparison = {}
        
        # Compare overall sentiment metrics
        current_overall = current_analysis.get('overall_sentiment', {})
        baseline_overall = baseline_analysis.get('overall_sentiment', {})
        
        for metric in ['vader_compound', 'textblob_polarity', 'textblob_subjectivity']:
            current_val = current_overall.get(metric, 0)
            baseline_val = baseline_overall.get(metric, 0)
            comparison[f'sentiment_{metric}_change'] = abs(current_val - baseline_val)
        
        # Compare manipulation markers
        current_manip = current_analysis.get('manipulation_markers', {})
        baseline_manip = baseline_analysis.get('manipulation_markers', {})
        
        for marker in self.manipulation_keywords.keys():
            ratio_key = f'{marker}_ratio'
            current_ratio = current_manip.get(ratio_key, 0)
            baseline_ratio = baseline_manip.get(ratio_key, 0)
            comparison[f'manipulation_{marker}_change'] = abs(current_ratio - baseline_ratio)
        
        # Compare complexity metrics
        current_complex = current_analysis.get('linguistic_complexity', {})
        baseline_complex = baseline_analysis.get('linguistic_complexity', {})
        
        for metric in ['avg_sentence_length', 'avg_word_length', 'syllable_complexity']:
            current_val = current_complex.get(metric, 0)
            baseline_val = baseline_complex.get(metric, 0)
            if baseline_val != 0:
                comparison[f'complexity_{metric}_change'] = abs(current_val - baseline_val) / baseline_val
            else:
                comparison[f'complexity_{metric}_change'] = abs(current_val)
        
        # Compare emotional volatility
        current_vol = current_analysis.get('emotional_volatility', {})
        baseline_vol = baseline_analysis.get('emotional_volatility', {})
        
        for metric in ['compound_volatility', 'polarity_volatility']:
            current_val = current_vol.get(metric, 0)
            baseline_val = baseline_vol.get(metric, 0)
            comparison[f'volatility_{metric}_change'] = abs(current_val - baseline_val)
        
        return comparison


if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Example usage
    sample_transcription = {
        'text': 'I feel really uncertain about this situation. Maybe we should be afraid of what everyone is saying.',
        'segments': [
            {'start': 0, 'end': 2, 'text': 'I feel really uncertain about this situation.'},
            {'start': 2, 'end': 4, 'text': 'Maybe we should be afraid of what everyone is saying.'}
        ],
        'audio_file': 'test.wav'
    }
    
    result = analyzer.analyze_sentiment(sample_transcription)
    print("Sentiment analysis completed:", bool(result))