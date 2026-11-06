"""
StoryBridge AI - FastAPI Backend Service
Provides RESTful endpoints for catalog retrieval, semantic vibe search, and cross-media recommendations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from typing import Optional, List

app = FastAPI(
    title="StoryBridge AI API",
    description="Cross-media semantic recommendation engine bridging books and movies.",
    version="1.0.0"
)

# Enable CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")

# In-memory catalog
with open(DATA_PATH, "r", encoding="utf-8") as f:
    CATALOG = json.load(f)

# Initialize recommender model lazily
_recommender = None


def get_recommender():
    global _recommender
    if _recommender is None:
        try:
            from ml.recommender import SemanticRecommender
            _recommender = SemanticRecommender()
        except Exception as e:
            print(f"ML Recommender initialization notice: {e}")
            _recommender = False
    return _recommender


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 8
    media_type: Optional[str] = None


@app.get("/")
def root():
    return {
        "service": "StoryBridge AI Recommendation Engine",
        "status": "online",
        "docs_url": "/docs",
        "catalog_size": len(CATALOG)
    }


@app.get("/api/catalog")
def get_catalog(type: Optional[str] = None, genre: Optional[str] = None):
    results = CATALOG
    if type:
        results = [item for item in results if item.get("type") == type]
    if genre:
        results = [item for item in results if genre.lower() in item.get("genre", "").lower()]
    return {"count": len(results), "items": results}


@app.post("/api/recommend")
def recommend_by_query(req: SearchRequest):
    recommender = get_recommender()
    if recommender:
        results = recommender.search_by_vibe(
            query=req.query,
            top_k=req.top_k,
            media_type=req.media_type
        )
        return {"query": req.query, "mode": "semantic_vector_search", "results": results}

    # Fallback to keyword matching if ML packages are not installed yet
    q = req.query.lower()
    words = q.split()
    matched = []
    for item in CATALOG:
        text = f"{item['title']} {item.get('author', '')} {item['genre']} {item['description']}".lower()
        if any(w in text for w in words):
            matched.append(item)
    return {"query": req.query, "mode": "keyword_fallback", "results": matched[:req.top_k]}


@app.get("/api/cross-media/{item_id}")
def cross_media(item_id: int, top_k: int = 5):
    recommender = get_recommender()
    if recommender:
        results = recommender.get_cross_media_recommendation(item_id=item_id, top_k=top_k)
        return {"item_id": item_id, "mode": "semantic_cross_media", "results": results}

    source = next((i for i in CATALOG if i["id"] == item_id), None)
    if not source:
        raise HTTPException(status_code=404, detail="Item not found")

    target_type = "movie" if source["type"] == "book" else "book"
    matches = [i for i in CATALOG if i["type"] == target_type and i["genre"] == source["genre"]]
    return {"item_id": item_id, "mode": "genre_fallback", "results": matches[:top_k]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

