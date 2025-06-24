# Deployment Guide

## Local Development

### Quick Start
```bash
# Install dependencies
pip install -r fastapi_requirements.txt

# Run the server
python fastapi_main.py
```

The API will be available at `http://localhost:8000`

## Docker Deployment

### Build and Run
```bash
# Build the image
docker build -t video-frame-processor .

# Run the container
docker run -p 8000:8000 video-frame-processor
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./frames:/app/frames
    environment:
      - LOG_LEVEL=info
```

## Production Deployment

### With External Qdrant

1. Set up Qdrant server:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Modify `fastapi_vector_db.py`:
```python
# Change in VectorDatabase.__init__
self.use_memory = False  # Use external Qdrant
```

### Environment Variables

- `LOG_LEVEL`: Set logging level (default: info)
- `MAX_UPLOAD_SIZE`: Maximum file size in bytes
- `QDRANT_HOST`: Qdrant server host
- `QDRANT_PORT`: Qdrant server port

### Performance Tuning

```bash
# Production server with multiple workers
uvicorn fastapi_main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn
gunicorn fastapi_main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Cloud Deployment

### AWS ECS/Fargate
1. Push Docker image to ECR
2. Create ECS task definition
3. Set up load balancer

### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/video-processor
gcloud run deploy --image gcr.io/PROJECT-ID/video-processor --platform managed
```

### Heroku
```bash
# Using Container Registry
heroku container:push web
heroku container:release web
```