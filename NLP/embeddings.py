"""
Vector embeddings module for creating semantic representations of speech content
to detect subtle changes in thought patterns and conceptual frameworks.
"""

import numpy as np
import json
from datetime import datetime
import pickle
import os

# Try to import sentence_transformers, fall back to basic embeddings if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence_transformers not available. Using basic text embeddings.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Limited embeddings functionality.")


class EmbeddingsAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize with a sentence transformer model for creating embeddings.
        Falls back to basic embeddings if sentence_transformers is not available.
        """
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.use_transformer = True
        else:
            # Fallback to basic embeddings
            self.model = None
            self.embedding_dim = 50  # Fixed dimension for basic embeddings
            self.use_transformer = False
            print("Using basic text embeddings (limited functionality)")
    
    def _create_basic_embedding(self, text):
        """
        Create basic embeddings using simple text features when transformers are not available.
        """
        if not text.strip():
            return np.zeros(self.embedding_dim)
        
        # Simple feature extraction
        words = text.lower().split()
        
        features = []
        
        # Length features
        features.append(len(text) / 1000.0)  # Text length normalized
        features.append(len(words) / 100.0)  # Word count normalized
        features.append(np.mean([len(w) for w in words]) if words else 0)  # Avg word length
        
        # Character frequency features (first 20 letters)
        char_counts = np.zeros(26)
        for char in text.lower():
            if 'a' <= char <= 'z':
                char_counts[ord(char) - ord('a')] += 1
        
        if len(text) > 0:
            char_counts = char_counts / len(text)  # Normalize by text length
        
        features.extend(char_counts[:20])  # First 20 letters
        
        # Common word features
        common_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 
                       'for', 'as', 'was', 'on', 'are', 'you', 'have', 'be', 'at', 'this',
                       'but', 'not', 'or', 'what', 'all']
        
        for word in common_words[:4]:  # Top 4 common words
            features.append(text.lower().count(word) / len(words) if words else 0)
        
        # Pad or truncate to embedding_dim
        features = features[:self.embedding_dim]
        while len(features) < self.embedding_dim:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
        
    def create_embeddings(self, transcription_data):
        """
        Create vector embeddings for the transcribed text and segments.
        """
        try:
            text = transcription_data.get('text', '')
            segments = transcription_data.get('segments', [])
            
            if not text.strip():
                return None
            
            # Overall text embedding
            if self.use_transformer:
                overall_embedding = self.model.encode([text])[0]
            else:
                overall_embedding = self._create_basic_embedding(text)
            
            # Segment embeddings
            segment_embeddings = []
            segment_texts = []
            
            for segment in segments:
                seg_text = segment.get('text', '').strip()
                if seg_text:
                    if self.use_transformer:
                        seg_embedding = self.model.encode([seg_text])[0]
                    else:
                        seg_embedding = self._create_basic_embedding(seg_text)
                        
                    segment_embeddings.append({
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'text': seg_text,
                        'embedding': seg_embedding.tolist()
                    })
                    segment_texts.append(seg_text)
            
            # Analyze semantic coherence across segments
            coherence_metrics = self._analyze_semantic_coherence(segment_embeddings)
            
            # Extract key concepts and themes
            concept_analysis = self._analyze_concepts(text, overall_embedding)
            
            # Analyze semantic drift within the speech
            semantic_drift = self._analyze_semantic_drift(segment_embeddings)
            
            embeddings_data = {
                'timestamp': datetime.now().isoformat(),
                'audio_file': transcription_data.get('audio_file', ''),
                'overall_embedding': overall_embedding.tolist(),
                'segment_embeddings': segment_embeddings,
                'semantic_coherence': coherence_metrics,
                'concept_analysis': concept_analysis,
                'semantic_drift': semantic_drift,
                'embedding_stats': {
                    'embedding_dimension': self.embedding_dim,
                    'num_segments': len(segment_embeddings),
                    'avg_segment_similarity': coherence_metrics.get('avg_similarity', 0)
                }
            }
            
            return embeddings_data
            
        except Exception as e:
            print(f"Embeddings analysis error: {str(e)}")
            return None
    
    def _analyze_semantic_coherence(self, segment_embeddings):
        """
        Analyze how semantically coherent the speech segments are.
        Sudden changes might indicate external influence.
        """
        if len(segment_embeddings) < 2:
            return {'avg_similarity': 1.0, 'min_similarity': 1.0, 'coherence_score': 1.0}
        
        embeddings = np.array([seg['embedding'] for seg in segment_embeddings])
        
        # Calculate pairwise similarities between consecutive segments
        consecutive_similarities = []
        for i in range(len(embeddings) - 1):
            if SKLEARN_AVAILABLE:
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            else:
                # Basic cosine similarity calculation
                a, b = embeddings[i], embeddings[i + 1]
                sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0 else 0
            consecutive_similarities.append(sim)
        
        # Calculate overall coherence metrics
        avg_similarity = np.mean(consecutive_similarities)
        min_similarity = np.min(consecutive_similarities)
        similarity_variance = np.var(consecutive_similarities)
        
        # Coherence score (higher = more coherent)
        coherence_score = avg_similarity * (1 - similarity_variance)
        
        return {
            'avg_similarity': float(avg_similarity),
            'min_similarity': float(min_similarity),
            'similarity_variance': float(similarity_variance),
            'coherence_score': float(coherence_score),
            'consecutive_similarities': [float(sim) for sim in consecutive_similarities]
        }
    
    def _analyze_concepts(self, text, embedding):
        """
        Analyze the main conceptual themes in the text.
        """
        # Split text into sentences for concept analysis
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {
                'num_concepts': 1,
                'concept_diversity': 0,
                'dominant_themes': []
            }
        
        # Create embeddings for each sentence
        if self.use_transformer:
            sentence_embeddings = self.model.encode(sentences)
        else:
            sentence_embeddings = np.array([self._create_basic_embedding(sent) for sent in sentences])
        
        # Use K-means clustering to identify distinct concepts
        n_clusters = min(3, len(sentences))  # Max 3 concept clusters
        
        if len(sentences) >= n_clusters and SKLEARN_AVAILABLE:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(sentence_embeddings)
            
            # Analyze concept distribution
            unique_clusters = len(set(cluster_labels))
            concept_diversity = unique_clusters / len(sentences)
            
            # Find dominant themes (most common clusters)
            cluster_counts = np.bincount(cluster_labels)
            dominant_clusters = np.argsort(cluster_counts)[::-1][:2]  # Top 2 themes
            
            dominant_themes = []
            for cluster_idx in dominant_clusters:
                # Find sentences in this cluster
                cluster_sentences = [sentences[i] for i, label in enumerate(cluster_labels) if label == cluster_idx]
                if cluster_sentences:
                    # Use the first sentence as theme representative
                    dominant_themes.append({
                        'theme_id': int(cluster_idx),
                        'representative_text': cluster_sentences[0][:100],  # First 100 chars
                        'sentence_count': len(cluster_sentences)
                    })
        else:
            # Fallback when sklearn is not available - treat each sentence as unique concept
            unique_clusters = len(sentences)
            concept_diversity = 1.0
            dominant_themes = [{'theme_id': 0, 'representative_text': sentences[0][:100], 'sentence_count': len(sentences)}]
        
        return {
            'num_concepts': unique_clusters,
            'concept_diversity': float(concept_diversity),
            'dominant_themes': dominant_themes,
            'total_sentences': len(sentences)
        }
    
    def _analyze_semantic_drift(self, segment_embeddings):
        """
        Analyze how much the semantic content drifts throughout the speech.
        """
        if len(segment_embeddings) < 3:
            return {'drift_score': 0, 'drift_direction': 'stable'}
        
        embeddings = np.array([seg['embedding'] for seg in segment_embeddings])
        
        if SKLEARN_AVAILABLE:
            # Calculate drift using PCA to find the main direction of change
            pca = PCA(n_components=1)
            pca.fit(embeddings)
            
            # Project embeddings onto the first principal component
            projected = pca.transform(embeddings).flatten()
            explained_variance_ratio = float(pca.explained_variance_ratio_[0])
        else:
            # Simple fallback - use first dimension as proxy for drift
            projected = embeddings[:, 0]
            explained_variance_ratio = 0.5  # Default value
        
        # Calculate drift score as the range of projection values
        drift_score = float(np.ptp(projected))  # Peak-to-peak range
        
        # Determine drift direction
        if projected[-1] > projected[0]:
            drift_direction = 'progressive'
        elif projected[-1] < projected[0]:
            drift_direction = 'regressive'
        else:
            drift_direction = 'stable'
        
        # Calculate drift rate (change per segment)
        drift_rate = abs(projected[-1] - projected[0]) / len(projected) if len(projected) > 1 else 0
        
        return {
            'drift_score': drift_score,
            'drift_direction': drift_direction,
            'drift_rate': float(drift_rate),
            'explained_variance_ratio': explained_variance_ratio
        }
    
    def compare_baseline(self, current_embeddings, baseline_embeddings):
        """
        Compare current embeddings against baseline to detect semantic shifts.
        """
        if not baseline_embeddings:
            return {}
        
        comparison = {}
        
        # Compare overall embeddings
        current_overall = np.array(current_embeddings.get('overall_embedding', []))
        baseline_overall = np.array(baseline_embeddings.get('overall_embedding', []))
        
        if len(current_overall) == len(baseline_overall) and len(current_overall) > 0:
            if SKLEARN_AVAILABLE:
                overall_similarity = cosine_similarity([current_overall], [baseline_overall])[0][0]
            else:
                # Basic cosine similarity calculation
                overall_similarity = np.dot(current_overall, baseline_overall) / (
                    np.linalg.norm(current_overall) * np.linalg.norm(baseline_overall)
                ) if np.linalg.norm(current_overall) > 0 and np.linalg.norm(baseline_overall) > 0 else 0
            
            comparison['overall_semantic_similarity'] = float(overall_similarity)
            comparison['overall_semantic_distance'] = float(1 - overall_similarity)
        
        # Compare coherence metrics
        current_coherence = current_embeddings.get('semantic_coherence', {})
        baseline_coherence = baseline_embeddings.get('semantic_coherence', {})
        
        coherence_change = abs(
            current_coherence.get('coherence_score', 0) - 
            baseline_coherence.get('coherence_score', 0)
        )
        comparison['coherence_change'] = float(coherence_change)
        
        # Compare concept diversity
        current_concepts = current_embeddings.get('concept_analysis', {})
        baseline_concepts = baseline_embeddings.get('concept_analysis', {})
        
        concept_diversity_change = abs(
            current_concepts.get('concept_diversity', 0) - 
            baseline_concepts.get('concept_diversity', 0)
        )
        comparison['concept_diversity_change'] = float(concept_diversity_change)
        
        # Compare semantic drift
        current_drift = current_embeddings.get('semantic_drift', {})
        baseline_drift = baseline_embeddings.get('semantic_drift', {})
        
        drift_change = abs(
            current_drift.get('drift_score', 0) - 
            baseline_drift.get('drift_score', 0)
        )
        comparison['semantic_drift_change'] = float(drift_change)
        
        return comparison
    
    def save_embeddings(self, embeddings_data, output_path):
        """
        Save embeddings data to file.
        """
        with open(output_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
    
    def load_embeddings(self, input_path):
        """
        Load embeddings data from file.
        """
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def create_baseline_profile(self, multiple_embeddings):
        """
        Create a baseline semantic profile from multiple recordings.
        """
        if not multiple_embeddings:
            return None
        
        # Average the overall embeddings
        overall_embeddings = [np.array(emb.get('overall_embedding', [])) for emb in multiple_embeddings]
        overall_embeddings = [emb for emb in overall_embeddings if len(emb) > 0]
        
        if overall_embeddings:
            baseline_overall = np.mean(overall_embeddings, axis=0)
        else:
            baseline_overall = np.zeros(self.embedding_dim)
        
        # Average coherence scores
        coherence_scores = [emb.get('semantic_coherence', {}).get('coherence_score', 0) for emb in multiple_embeddings]
        baseline_coherence = np.mean(coherence_scores) if coherence_scores else 0
        
        # Average concept diversity
        concept_diversities = [emb.get('concept_analysis', {}).get('concept_diversity', 0) for emb in multiple_embeddings]
        baseline_concept_diversity = np.mean(concept_diversities) if concept_diversities else 0
        
        # Average drift scores
        drift_scores = [emb.get('semantic_drift', {}).get('drift_score', 0) for emb in multiple_embeddings]
        baseline_drift = np.mean(drift_scores) if drift_scores else 0
        
        baseline_profile = {
            'timestamp': datetime.now().isoformat(),
            'overall_embedding': baseline_overall.tolist(),
            'semantic_coherence': {'coherence_score': float(baseline_coherence)},
            'concept_analysis': {'concept_diversity': float(baseline_concept_diversity)},
            'semantic_drift': {'drift_score': float(baseline_drift)},
            'num_recordings': len(multiple_embeddings)
        }
        
        return baseline_profile


if __name__ == "__main__":
    # Test the embeddings analyzer
    analyzer = EmbeddingsAnalyzer()
    
    # Example usage
    sample_transcription = {
        'text': 'This is a test of the semantic analysis system. We want to detect changes in thinking patterns.',
        'segments': [
            {'start': 0, 'end': 2, 'text': 'This is a test of the semantic analysis system.'},
            {'start': 2, 'end': 4, 'text': 'We want to detect changes in thinking patterns.'}
        ],
        'audio_file': 'test.wav'
    }
    
    result = analyzer.create_embeddings(sample_transcription)
    print("Embeddings analysis completed:", bool(result))