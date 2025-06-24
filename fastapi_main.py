"""
FastAPI Video Frame Processor

A FastAPI application for video processing that extracts frames from uploaded videos,
computes feature vectors, and provides similarity search capabilities using Qdrant vector database.
"""

import os
import logging
import uuid
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from fastapi_video_processor import VideoProcessor
from fastapi_vector_db import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Global instances
video_processor = None
vector_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    global video_processor, vector_db
    
    # Startup
    logger.info("Initializing video processor and vector database...")
    video_processor = VideoProcessor()
    vector_db = VectorDatabase()
    await vector_db.initialize()
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    if vector_db:
        await vector_db.close()
    logger.info("Application shutdown complete")

# FastAPI app
app = FastAPI(
    title="Video Frame Processor API",
    description="Extract frames from videos, compute feature vectors, and search for similar content",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class VideoUploadResponse(BaseModel):
    success: bool
    video_id: str
    frames_count: int
    frames: List[dict]
    message: str

class SimilaritySearchRequest(BaseModel):
    frame_id: str
    top_k: int = Field(default=5, ge=1, le=20)

class SimilaritySearchResponse(BaseModel):
    query_frame_id: str
    similar_frames: List[dict]

class FrameInfo(BaseModel):
    id: str
    timestamp: float
    frame_number: int
    video_id: str
    similarity: Optional[float] = None

class VideoFramesResponse(BaseModel):
    video_id: str
    frames: List[FrameInfo]

class HealthResponse(BaseModel):
    status: str
    version: str
    database_status: str

# Dependency functions
def get_video_processor():
    """Get video processor instance."""
    if video_processor is None:
        raise HTTPException(status_code=500, detail="Video processor not initialized")
    return video_processor

def get_vector_db():
    """Get vector database instance."""
    if vector_db is None:
        raise HTTPException(status_code=500, detail="Vector database not initialized")
    return vector_db

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Video Frame Processor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(db: VectorDatabase = Depends(get_vector_db)):
    """Health check endpoint."""
    try:
        db_status = "connected" if await db.is_connected() else "disconnected"
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            database_status=db_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            database_status="error"
        )

@app.post("/api/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    frame_interval: float = Form(default=1.0, ge=0.1, le=60.0),
    processor: VideoProcessor = Depends(get_video_processor),
    db: VectorDatabase = Depends(get_vector_db)
):
    """
    Upload and process a video file.
    
    - **video**: Video file to upload (MP4, AVI, MOV, MKV)
    - **frame_interval**: Time interval between extracted frames in seconds
    """
    try:
        # Validate file
        if not video.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not allowed_file(video.filename):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Allowed: MP4, AVI, MOV, MKV"
            )
        
        # Check file size
        content = await video.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")
        
        # Generate unique identifiers
        video_id = str(uuid.uuid4())
        file_extension = video.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{video_id}.{file_extension}"
        
        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        with open(filepath, 'wb') as f:
            f.write(content)
        
        logger.info(f"Processing video: {unique_filename}")
        
        # Extract frames
        frames_info = processor.extract_frames(
            filepath, FRAMES_FOLDER, video_id, frame_interval
        )
        
        if not frames_info:
            raise HTTPException(status_code=500, detail="Failed to extract frames from video")
        
        # Process frames in background
        background_tasks.add_task(
            process_frames_background, frames_info, processor, db
        )
        
        # Return immediate response
        frame_list = [
            {
                'id': frame['id'],
                'timestamp': frame['timestamp'],
                'frame_number': frame['frame_number']
            }
            for frame in frames_info
        ]
        
        return VideoUploadResponse(
            success=True,
            video_id=video_id,
            frames_count=len(frames_info),
            frames=frame_list,
            message=f"Successfully uploaded video. Processing {len(frames_info)} frames..."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video upload: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def process_frames_background(
    frames_info: List[dict], 
    processor: VideoProcessor, 
    db: VectorDatabase
):
    """Background task to compute feature vectors and store in database."""
    try:
        logger.info(f"Computing feature vectors for {len(frames_info)} frames")
        
        for frame_info in frames_info:
            feature_vector = processor.compute_feature_vector(frame_info['path'])
            if feature_vector is not None:
                await db.add_vector(
                    frame_info['id'],
                    feature_vector.tolist(),
                    {
                        'video_id': frame_info['video_id'],
                        'frame_path': frame_info['path'],
                        'timestamp': frame_info['timestamp'],
                        'frame_number': frame_info['frame_number']
                    }
                )
        
        logger.info(f"Completed processing {len(frames_info)} frames")
        
    except Exception as e:
        logger.error(f"Error in background processing: {e}")

@app.post("/api/search", response_model=SimilaritySearchResponse)
async def search_similar_frames(
    request: SimilaritySearchRequest,
    db: VectorDatabase = Depends(get_vector_db)
):
    """
    Search for similar frames based on a given frame ID.
    
    - **frame_id**: ID of the frame to use as query
    - **top_k**: Number of similar frames to return (1-20)
    """
    try:
        # Get the feature vector for the query frame
        query_vector = await db.get_vector(request.frame_id)
        if query_vector is None:
            raise HTTPException(status_code=404, detail="Frame not found")
        
        # Search for similar frames
        similar_frames = await db.search_similar(query_vector, request.top_k + 1)
        
        # Remove the query frame from results
        similar_frames = [f for f in similar_frames if f['id'] != request.frame_id][:request.top_k]
        
        return SimilaritySearchResponse(
            query_frame_id=request.frame_id,
            similar_frames=similar_frames
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/frames/{video_id}", response_model=VideoFramesResponse)
async def get_video_frames(
    video_id: str,
    db: VectorDatabase = Depends(get_vector_db)
):
    """
    Get all frames for a specific video.
    
    - **video_id**: Unique identifier of the video
    """
    try:
        frames = await db.get_frames_by_video(video_id)
        
        frame_list = [
            FrameInfo(
                id=frame['id'],
                timestamp=frame['timestamp'],
                frame_number=frame['frame_number'],
                video_id=frame['video_id']
            )
            for frame in frames
        ]
        
        return VideoFramesResponse(
            video_id=video_id,
            frames=frame_list
        )
        
    except Exception as e:
        logger.error(f"Error getting frames for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get frames: {str(e)}")

@app.get("/frame/{frame_id}")
async def get_frame_image(
    frame_id: str,
    db: VectorDatabase = Depends(get_vector_db)
):
    """
    Retrieve frame image by ID.
    
    - **frame_id**: Unique identifier of the frame
    """
    try:
        frame_data = await db.get_frame_metadata(frame_id)
        if not frame_data or 'frame_path' not in frame_data:
            raise HTTPException(status_code=404, detail="Frame not found")
        
        frame_path = frame_data['frame_path']
        if not os.path.exists(frame_path):
            raise HTTPException(status_code=404, detail="Frame file not found")
        
        return FileResponse(
            frame_path,
            media_type='image/jpeg',
            filename=f"frame_{frame_id}.jpg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving frame {frame_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve frame: {str(e)}")

@app.get("/api/stats")
async def get_database_stats(db: VectorDatabase = Depends(get_vector_db)):
    """Get database statistics."""
    try:
        stats = await db.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )