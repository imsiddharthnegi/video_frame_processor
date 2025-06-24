import numpy as np
import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class VectorDB:
    """In-memory vector database for storing and searching feature vectors."""
    
    def __init__(self, metadata_file: str = 'vector_metadata.json'):
        self.vectors = {}  # id -> numpy array
        self.metadata = {}  # id -> metadata dict
        self.metadata_file = metadata_file
        self.logger = logging.getLogger(__name__)
        
        # Load existing data if available
        self.load_metadata()
    
    def add_vector(self, vector_id: str, vector: np.ndarray, metadata: Dict) -> bool:
        """
        Add a feature vector with metadata to the database.
        
        Args:
            vector_id: Unique identifier for the vector
            vector: Feature vector as numpy array
            metadata: Associated metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vectors[vector_id] = vector.copy()
            self.metadata[vector_id] = {
                **metadata,
                'created_at': datetime.now().isoformat(),
                'vector_size': len(vector)
            }
            
            # Save metadata to file
            self.save_metadata()
            
            self.logger.debug(f"Added vector {vector_id} with {len(vector)} dimensions")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding vector {vector_id}: {str(e)}")
            return False
    
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a vector by ID.
        
        Args:
            vector_id: Vector identifier
            
        Returns:
            Vector as numpy array or None if not found
        """
        return self.vectors.get(vector_id)
    
    def get_frame_metadata(self, vector_id: str) -> Optional[Dict]:
        """
        Get metadata for a frame/vector.
        
        Args:
            vector_id: Vector identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self.metadata.get(vector_id)
    
    def search_similar(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query feature vector
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with similar vectors and their metadata
        """
        try:
            if len(self.vectors) == 0:
                return []
            
            similarities = []
            
            # Calculate cosine similarity with all vectors
            for vector_id, stored_vector in self.vectors.items():
                similarity = self.cosine_similarity(query_vector, stored_vector)
                similarities.append({
                    'id': vector_id,
                    'similarity': similarity,
                    'metadata': self.metadata.get(vector_id, {})
                })
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top k results
            results = similarities[:top_k]
            
            self.logger.debug(f"Found {len(results)} similar vectors for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching similar vectors: {str(e)}")
            return []
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Handle zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            similarity = dot_product / (norm1 * norm2)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def get_frames_by_video(self, video_id: str) -> List[Dict]:
        """
        Get all frames belonging to a specific video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of frame metadata dictionaries
        """
        try:
            frames = []
            for vector_id, metadata in self.metadata.items():
                if metadata.get('video_id') == video_id:
                    frames.append({
                        'id': vector_id,
                        'timestamp': metadata.get('timestamp', 0),
                        'frame_number': metadata.get('frame_number', 0),
                        'frame_path': metadata.get('frame_path', ''),
                        'created_at': metadata.get('created_at', ''),
                        'vector_size': metadata.get('vector_size', 0)
                    })
            
            # Sort by timestamp
            frames.sort(key=lambda x: x['timestamp'])
            
            self.logger.debug(f"Found {len(frames)} frames for video {video_id}")
            return frames
            
        except Exception as e:
            self.logger.error(f"Error getting frames for video {video_id}: {str(e)}")
            return []
    
    def search_by_video(self, video_id: str, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar frames within a specific video.
        
        Args:
            video_id: Video identifier to search within
            query_vector: Query feature vector
            top_k: Number of top results to return
            
        Returns:
            List of similar frames within the video
        """
        try:
            video_vectors = {}
            
            # Filter vectors by video ID
            for vector_id, metadata in self.metadata.items():
                if metadata.get('video_id') == video_id and vector_id in self.vectors:
                    video_vectors[vector_id] = self.vectors[vector_id]
            
            if not video_vectors:
                return []
            
            similarities = []
            
            # Calculate similarities within the video
            for vector_id, stored_vector in video_vectors.items():
                similarity = self.cosine_similarity(query_vector, stored_vector)
                similarities.append({
                    'id': vector_id,
                    'similarity': similarity,
                    'metadata': self.metadata.get(vector_id, {})
                })
            
            # Sort and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error searching within video {video_id}: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            video_counts = {}
            total_vectors = len(self.vectors)
            
            # Count vectors per video
            for metadata in self.metadata.values():
                video_id = metadata.get('video_id', 'unknown')
                video_counts[video_id] = video_counts.get(video_id, 0) + 1
            
            return {
                'total_vectors': total_vectors,
                'total_videos': len(video_counts),
                'vectors_per_video': video_counts,
                'average_vectors_per_video': total_vectors / len(video_counts) if video_counts else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def save_metadata(self) -> bool:
        """
        Save metadata to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a serializable version of metadata
            serializable_metadata = {}
            for vector_id, metadata in self.metadata.items():
                serializable_metadata[vector_id] = {
                    k: v for k, v in metadata.items()
                    if not isinstance(v, np.ndarray)  # Skip numpy arrays
                }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
            return False
    
    def load_metadata(self) -> bool:
        """
        Load metadata from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.metadata_file):
                self.logger.info("No existing metadata file found")
                return True
            
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            self.logger.info(f"Loaded metadata for {len(self.metadata)} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading metadata: {str(e)}")
            return False
    
    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector and its metadata.
        
        Args:
            vector_id: Vector identifier to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if vector_id in self.vectors:
                del self.vectors[vector_id]
            
            if vector_id in self.metadata:
                del self.metadata[vector_id]
            
            self.save_metadata()
            
            self.logger.debug(f"Deleted vector {vector_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting vector {vector_id}: {str(e)}")
            return False
    
    def clear_database(self) -> bool:
        """
        Clear all vectors and metadata.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vectors.clear()
            self.metadata.clear()
            self.save_metadata()
            
            self.logger.info("Cleared all vectors and metadata")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing database: {str(e)}")
            return False
