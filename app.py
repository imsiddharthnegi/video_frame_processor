from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import cv2
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

app = FastAPI()

FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

COLLECTION_NAME = "frames"
VECTOR_SIZE = 256

client = QdrantClient(
    url="https://71d0f16d-b032-432e-8aa2-5dd6b7afee4d.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TkLAfO_fT2c7mzMnbgW9qU5YpkA-1W5IODdWFhUkqQY"
)

if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(count % (fps * interval)) == 0:
            frame_path = os.path.join(FRAME_DIR, f"frame_{idx}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            idx += 1
        count += 1
    cap.release()
    return frames

def compute_feature_vector(image_path):
    img = cv2.imread(image_path)
    hist = cv2.calcHist([img], [0,1,2], None, [4,4,4], [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    # Pad or trim to VECTOR_SIZE
    if len(hist) > VECTOR_SIZE:
        hist = hist[:VECTOR_SIZE]
    elif len(hist) < VECTOR_SIZE:
        hist = np.pad(hist, (0, VECTOR_SIZE - len(hist)))
    return hist.tolist()

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4", "video/mkv", "video/avi"]:
        raise HTTPException(status_code=400, detail="Invalid video type.")
    video_path = f"tmp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())
    frames = extract_frames(video_path)
    points = []
    for idx, frame_path in enumerate(frames):
        vector = compute_feature_vector(frame_path)
        points.append(PointStruct(id=idx, vector=vector, payload={"frame_path": frame_path}))
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    os.remove(video_path)
    return {"frames_saved": len(frames)}

@app.post("/search/")
async def search_similar(vector: List[float]):
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=5
    )
    results = []
    for hit in search_result:
        frame_path = hit.payload["frame_path"]
        results.append({
            "frame": frame_path,
            "vector": hit.vector
        })
    return results

@app.get("/frame/{frame_id}")
def get_frame(frame_id: int):
    frame_path = os.path.join(FRAME_DIR, f"frame_{frame_id}.jpg")
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(frame_path, media_type="image/jpeg")
