"""
FastAPI Vector Database Module

Provides vector database functionality using Qdrant for storing and searching feature vectors
in the FastAPI video frame processing application.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse

class VectorDatabase:
    """Vector database using Qdrant for storing and searching feature vectors."""
    
    def __init__(self, collection_name: str = "video_frames", use_memory: bool = True):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the Qdrant collection
            use_memory: Whether to use in-memory Qdrant (True) or connect to external instance
        """
        self.collection_name = collection_name
        self.client = None
        self.use_memory = use_memory
        self.logger = logging.getLogger(__name__)
        self.vector_size = 202  # Size of our feature vectors
        
    async def initialize(self):
        """Initialize the Qdrant client and create collection if needed."""
        try:
            if self.use_memory:
                # Use in-memory Qdrant for simplicity
                self.client = QdrantClient(":memory:")
                self.logger.info("Initialized in-memory Qdrant client")
            else:
                # Connect to external Qdrant instance
                # You can modify this to connect to your Qdrant server
                self.client = QdrantClient(url="http://localhost:6333")
                self.logger.info("Connected to external Qdrant instance")
            
            # Create collection if it doesn't exist
            await self._create_collection()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    async def _create_collection(self):
        """Create the collection with appropriate vector configuration."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created collection '{self.collection_name}'")
            else:
                self.logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise
    
    async def add_vector(self, vector_id: str, vector: List[float], metadata: Dict) -> bool:
        """
        Add a feature vector with metadata to the database.
        
        Args:
            vector_id: Unique identifier for the vector
            vector: Feature vector as list of floats
            metadata: Associated metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure vector is the correct size
            if len(vector) != self.vector_size:
                self.logger.error(f"Vector size mismatch: expected {self.vector_size}, got {len(vector)}")
                return False
            
            # Add timestamp to metadata
            metadata_with_timestamp = {
                **metadata,
                'created_at': datetime.now().isoformat(),
                'vector_size': len(vector)
            }
            
            # Create point
            point = PointStruct(
                id=vector_id,
                vector=vector,
                payload=metadata_with_timestamp
            )
            
            # Insert point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            self.logger.debug(f"Added vector {vector_id} with {len(vector)} dimensions")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding vector {vector_id}: {e}")
            return False
    
    async def get_vector(self, vector_id: str) -> Optional[List[float]]:
        """
        Retrieve a vector by ID.
        
        Args:
            vector_id: Vector identifier
            
        Returns:
            Vector as list of floats or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id],
                with_vectors=True
            )
            
            if result and len(result) > 0:
                return result[0].vector
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving vector {vector_id}: {e}")
            return None
    
    async def get_frame_metadata(self, vector_id: str) -> Optional[Dict]:
        """
        Get metadata for a frame/vector.
        
        Args:
            vector_id: Vector identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id],
                with_payload=True
            )
            
            if result and len(result) > 0:
                return result[0].payload
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving metadata for {vector_id}: {e}")
            return None
    
    async def search_similar(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query feature vector
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with similar vectors and their metadata
        """
        try:
            if len(query_vector) != self.vector_size:
                self.logger.error(f"Query vector size mismatch: expected {self.vector_size}, got {len(query_vector)}")
                return []
            
            # Search for similar vectors
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    'id': point.id,
                    'similarity': point.score,
                    'metadata': point.payload
                })
            
            self.logger.debug(f"Found {len(results)} similar vectors")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching similar vectors: {e}")
            return []
    
    async def get_frames_by_video(self, video_id: str) -> List[Dict]:
        """
        Get all frames belonging to a specific video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of frame metadata dictionaries
        """
        try:
            # Search with filter for specific video_id
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="video_id",
                            match=MatchValue(value=video_id)
                        )
                    ]
                ),
                with_payload=True,
                limit=1000  # Adjust based on expected frame count
            )
            
            # Format results
            frames = []
            for point in search_result[0]:  # scroll returns (points, next_page_offset)
                payload = point.payload
                frames.append({
                    'id': point.id,
                    'timestamp': payload.get('timestamp', 0),
                    'frame_number': payload.get('frame_number', 0),
                    'frame_path': payload.get('frame_path', ''),
                    'video_id': payload.get('video_id', ''),
                    'created_at': payload.get('created_at', ''),
                    'vector_size': payload.get('vector_size', 0)
                })
            
            # Sort by timestamp
            frames.sort(key=lambda x: x['timestamp'])
            
            self.logger.debug(f"Found {len(frames)} frames for video {video_id}")
            return frames
            
        except Exception as e:
            self.logger.error(f"Error getting frames for video {video_id}: {e}")
            return []
    
    async def search_by_video(self, video_id: str, query_vector: List[float], top_k: int = 5) -> List[Dict]:
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
            if len(query_vector) != self.vector_size:
                self.logger.error(f"Query vector size mismatch: expected {self.vector_size}, got {len(query_vector)}")
                return []
            
            # Search with filter for specific video_id
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="video_id",
                            match=MatchValue(value=video_id)
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    'id': point.id,
                    'similarity': point.score,
                    'metadata': point.payload
                })
            
            self.logger.debug(f"Found {len(results)} similar frames in video {video_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching within video {video_id}: {e}")
            return []
    
    async def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get all points to calculate video statistics
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=10000  # Adjust based on expected total count
            )[0]
            
            # Count vectors per video
            video_counts = {}
            for point in all_points:
                video_id = point.payload.get('video_id', 'unknown')
                video_counts[video_id] = video_counts.get(video_id, 0) + 1
            
            total_vectors = collection_info.points_count
            
            return {
                'total_vectors': total_vectors,
                'total_videos': len(video_counts),
                'vectors_per_video': video_counts,
                'average_vectors_per_video': total_vectors / len(video_counts) if video_counts else 0,
                'collection_status': collection_info.status.value if hasattr(collection_info.status, 'value') else str(collection_info.status)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector and its metadata.
        
        Args:
            vector_id: Vector identifier to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[vector_id]
            )
            
            self.logger.debug(f"Deleted vector {vector_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting vector {vector_id}: {e}")
            return False
    
    async def clear_database(self) -> bool:
        """
        Clear all vectors and metadata.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            await self._create_collection()
            
            self.logger.info("Cleared all vectors and metadata")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            return False
    
    async def is_connected(self) -> bool:
        """
        Check if the database is connected and accessible.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            if self.client is None:
                return False
            
            # Try to get collection info
            self.client.get_collection(self.collection_name)
            return True
            
        except Exception as e:
            self.logger.error(f"Database connection check failed: {e}")
            return False
    
    async def close(self):
        """Close the database connection."""
        try:
            if self.client:
                self.client.close()
                self.logger.info("Closed vector database connection")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")