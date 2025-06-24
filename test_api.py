"""
Test script for the FastAPI Video Frame Processor

This script demonstrates how to use the API endpoints and provides
example usage for all main functionality.
"""

import requests
import json
import time
import os
from typing import Optional

class VideoProcessorAPIClient:
    """Client for testing the Video Frame Processor API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def upload_video(self, video_path: str, frame_interval: float = 1.0) -> dict:
        """Upload a video for processing."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            data = {'frame_interval': frame_interval}
            
            response = self.session.post(
                f"{self.base_url}/api/upload",
                files=files,
                data=data
            )
            
        return response.json()
    
    def search_similar(self, frame_id: str, top_k: int = 5) -> dict:
        """Search for similar frames."""
        data = {
            "frame_id": frame_id,
            "top_k": top_k
        }
        
        response = self.session.post(
            f"{self.base_url}/api/search",
            json=data
        )
        
        return response.json()
    
    def get_video_frames(self, video_id: str) -> dict:
        """Get all frames for a video."""
        response = self.session.get(f"{self.base_url}/api/frames/{video_id}")
        return response.json()
    
    def get_frame_image(self, frame_id: str, save_path: Optional[str] = None) -> bytes:
        """Download frame image."""
        response = self.session.get(f"{self.base_url}/frame/{frame_id}")
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        
        return response.content
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        response = self.session.get(f"{self.base_url}/api/stats")
        return response.json()

def test_api_workflow():
    """Test the complete API workflow."""
    client = VideoProcessorAPIClient()
    
    print("🔍 Testing Video Frame Processor API")
    print("=" * 50)
    
    # 1. Health check
    print("\n1. Health Check")
    try:
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"📊 Database: {health['database_status']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # 2. Get initial stats
    print("\n2. Initial Database Stats")
    try:
        stats = client.get_stats()
        print(f"📈 Total vectors: {stats.get('total_vectors', 0)}")
        print(f"🎬 Total videos: {stats.get('total_videos', 0)}")
    except Exception as e:
        print(f"⚠️ Stats unavailable: {e}")
    
    # 3. Video upload (you need to provide a test video)
    print("\n3. Video Upload Test")
    test_video_path = "test_video.mp4"  # You need to provide this
    
    if os.path.exists(test_video_path):
        try:
            print(f"📤 Uploading video: {test_video_path}")
            upload_result = client.upload_video(test_video_path, frame_interval=2.0)
            
            if upload_result.get('success'):
                video_id = upload_result['video_id']
                frames_count = upload_result['frames_count']
                print(f"✅ Upload successful!")
                print(f"🆔 Video ID: {video_id}")
                print(f"🖼️ Frames extracted: {frames_count}")
                
                # Wait for processing to complete
                print("⏳ Waiting for background processing...")
                time.sleep(5)
                
                # 4. Get video frames
                print("\n4. Getting Video Frames")
                frames_result = client.get_video_frames(video_id)
                frames = frames_result.get('frames', [])
                print(f"📋 Retrieved {len(frames)} frames")
                
                if frames:
                    # 5. Download a frame image
                    print("\n5. Downloading Frame Image")
                    first_frame = frames[0]
                    frame_id = first_frame['id']
                    save_path = f"test_frame_{frame_id}.jpg"
                    
                    try:
                        client.get_frame_image(frame_id, save_path)
                        print(f"💾 Frame saved: {save_path}")
                    except Exception as e:
                        print(f"❌ Frame download failed: {e}")
                    
                    # 6. Similarity search
                    print("\n6. Similarity Search")
                    try:
                        search_result = client.search_similar(frame_id, top_k=3)
                        similar_frames = search_result.get('similar_frames', [])
                        print(f"🔍 Found {len(similar_frames)} similar frames")
                        
                        for i, frame in enumerate(similar_frames[:3]):
                            similarity = frame.get('similarity', 0)
                            print(f"   {i+1}. Frame {frame['id']}: {similarity:.3f} similarity")
                    
                    except Exception as e:
                        print(f"❌ Similarity search failed: {e}")
            
            else:
                print(f"❌ Upload failed: {upload_result}")
        
        except Exception as e:
            print(f"❌ Upload error: {e}")
    else:
        print(f"⚠️ Test video not found: {test_video_path}")
        print("   Place a test video file named 'test_video.mp4' in the current directory")
    
    # 7. Final stats
    print("\n7. Final Database Stats")
    try:
        final_stats = client.get_stats()
        print(f"📈 Total vectors: {final_stats.get('total_vectors', 0)}")
        print(f"🎬 Total videos: {final_stats.get('total_videos', 0)}")
        
        videos_per_video = final_stats.get('vectors_per_video', {})
        if videos_per_video:
            print("📊 Vectors per video:")
            for vid, count in videos_per_video.items():
                print(f"   {vid[:8]}...: {count} vectors")
    
    except Exception as e:
        print(f"⚠️ Final stats unavailable: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API test completed!")

def create_sample_curl_commands():
    """Generate sample cURL commands for testing."""
    
    commands = """
# Sample cURL Commands for Video Frame Processor API

## 1. Health Check
curl -X GET "http://localhost:8000/health"

## 2. Upload Video
curl -X POST "http://localhost:8000/api/upload" \\
  -F "video=@test_video.mp4" \\
  -F "frame_interval=1.0"

## 3. Search Similar Frames
curl -X POST "http://localhost:8000/api/search" \\
  -H "Content-Type: application/json" \\
  -d '{
    "frame_id": "VIDEO_ID_0001",
    "top_k": 5
  }'

## 4. Get Video Frames
curl -X GET "http://localhost:8000/api/frames/VIDEO_ID"

## 5. Download Frame Image
curl -X GET "http://localhost:8000/frame/FRAME_ID" -o frame.jpg

## 6. Get Database Stats
curl -X GET "http://localhost:8000/api/stats"

## 7. API Documentation
# Visit http://localhost:8000/docs for interactive documentation
"""
    
    return commands

if __name__ == "__main__":
    print("Video Frame Processor API Test Client")
    print("=====================================")
    
    # Generate sample commands
    print("\n📋 Sample cURL Commands:")
    print(create_sample_curl_commands())
    
    # Run API tests
    print("\n🧪 Running API Tests:")
    test_api_workflow()