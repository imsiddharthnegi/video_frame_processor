# Video Frame Processor FastAPI

A FastAPI application for video processing that extracts frames from uploaded videos, computes feature vectors, and provides similarity search capabilities using Qdrant vector database.

## Features

- **Video Upload & Processing**: Upload videos (MP4, AVI, MOV, MKV) and extract frames at configurable intervals
- **Feature Vector Computation**: Compute color histogram and texture-based feature vectors for each frame
- **Vector Database**: Store feature vectors in Qdrant vector database for efficient similarity search
- **Similarity Search**: Find visually similar frames using cosine similarity
- **RESTful API**: Complete REST API with automatic documentation

## Architecture

### Core Components

- **FastAPI Application** (`fastapi_main.py`): Main application with REST API endpoints
- **Video Processor** (`fastapi_video_processor.py`): Handles video frame extraction and feature computation
- **Vector Database** (`fastapi_vector_db.py`): Qdrant-based vector storage and similarity search

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint with API information |
| GET | `/health` | Health check endpoint |
| POST | `/api/upload` | Upload and process video file |
| POST | `/api/search` | Search for similar frames |
| GET | `/api/frames/{video_id}` | Get all frames for a video |
| GET | `/frame/{frame_id}` | Retrieve frame image |
| GET | `/api/stats` | Get database statistics |

## Installation

### Prerequisites

- Python 3.8+
- OpenCV dependencies (automatically installed)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd video-frame-processor-fastapi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create required directories:
```bash
mkdir uploads frames
```

## Usage

### Start the Server

```bash
python fastapi_main.py
```

The server will start on `http://0.0.0.0:8000`

### API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Example Usage

#### 1. Upload a Video

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "video=@your_video.mp4" \
  -F "frame_interval=1.0"
```

Response:
```json
{
  "success": true,
  "video_id": "uuid-here",
  "frames_count": 30,
  "frames": [...],
  "message": "Successfully uploaded video. Processing 30 frames..."
}
```

#### 2. Search for Similar Frames

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "frame_id": "uuid_0001",
    "top_k": 5
  }'
```

Response:
```json
{
  "query_frame_id": "uuid_0001",
  "similar_frames": [
    {
      "id": "uuid_0005",
      "similarity": 0.95,
      "metadata": {...}
    }
  ]
}
```

#### 3. Get Frame Image

```bash
curl "http://localhost:8000/frame/uuid_0001" -o frame.jpg
```

#### 4. Get Video Frames

```bash
curl "http://localhost:8000/api/frames/video-uuid"
```

## Configuration

### Video Processing
- **Supported formats**: MP4, AVI, MOV, MKV
- **Max file size**: 100MB
- **Frame interval**: 0.1-60.0 seconds
- **Image quality**: JPEG with 85% quality
- **Max resolution**: Automatically resized to 1920x1080 if larger

### Feature Vectors
- **Size**: 202 dimensions
- **Components**:
  - Color histograms (192 features): RGB channels with 64 bins each
  - Color statistics (6 features): Mean and standard deviation for RGB
  - Texture features (4 features): Gradient magnitude statistics

### Vector Database
- **Engine**: Qdrant (in-memory by default)
- **Distance metric**: Cosine similarity
- **Collection**: `video_frames`

## Development

### Project Structure

```
├── fastapi_main.py              # Main FastAPI application
├── fastapi_video_processor.py   # Video processing module
├── fastapi_vector_db.py         # Vector database module
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── uploads/                     # Video upload directory
└── frames/                      # Extracted frames directory
```

### Running in Development Mode

```bash
uvicorn fastapi_main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing

```bash
# Health check
curl http://localhost:8000/health

# API stats
curl http://localhost:8000/api/stats
```

## Deployment

### Production Settings

For production deployment, consider:

1. **External Qdrant**: Connect to external Qdrant instance
2. **File Storage**: Use cloud storage for video/frame files
3. **Load Balancing**: Use multiple worker processes
4. **Monitoring**: Add application monitoring and logging

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p uploads frames

EXPOSE 8000
CMD ["uvicorn", "fastapi_main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Performance Considerations

- **Background Processing**: Feature computation runs in background to provide immediate upload response
- **Memory Usage**: In-memory Qdrant suitable for development; use external instance for production
- **Concurrent Uploads**: FastAPI handles concurrent requests efficiently
- **Frame Caching**: Extracted frames are cached on disk

## Error Handling

The API includes comprehensive error handling:
- File validation (type, size)
- Video processing errors
- Database connection issues
- Missing resources (404)
- Server errors (500)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.