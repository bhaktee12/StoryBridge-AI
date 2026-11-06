"""
StoryBridge AI - ML Recommender Engine
Uses Sentence-Transformers to compute semantic vector embeddings across book blurbs
and movie synopses, enabling cross-media recommendations via Cosine Similarity.
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "embeddings_cache.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticRecommender:
    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path
        self.df = self._load_data()
        self.model = None
        self.embeddings = None
        self._initialize_embeddings()

    def _load_data(self) -> pd.DataFrame:
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _initialize_embeddings(self):
        # Check if cached embeddings exist
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "rb") as f:
                    cached = pickle.load(f)
                    self.embeddings = cached["embeddings"]
                    print(f"Loaded {len(self.embeddings)} precomputed embeddings from cache.")
                    return
            except Exception:
                pass

        print(f"Loading transformer model: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME)
        descriptions = self.df["description"].fillna("").tolist()
        print(f"Generating embeddings for {len(descriptions)} catalog items...")
        self.embeddings = self.model.encode(descriptions, show_progress_bar=False)

        try:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump({"embeddings": self.embeddings}, f)
            print("Embeddings cached successfully.")
        except Exception as e:
            print(f"Cache save warning: {e}")

    def _get_model(self):
        if self.model is None:
            self.model = SentenceTransformer(MODEL_NAME)
        return self.model

    def search_by_vibe(self, query: str, top_k: int = 6, media_type: str = None) -> list:
        """
        Takes a natural language prompt (e.g. 'isolated space station mystery')
        and returns the top-k semantically matched items.
        """
        model = self._get_model()
        query_vector = model.encode([query])
        similarities = cosine_similarity(query_vector, self.embeddings)[0]

        ranked_indices = similarities.argsort()[::-1]
        results = []

        for idx in ranked_indices:
            row = self.df.iloc[idx].to_dict()
            if media_type and row.get("type") != media_type:
                continue

            score = float(similarities[idx])
            row["match_score"] = round(score * 100, 1)
            results.append(row)

            if len(results) >= top_k:
                break

        return results

    def get_cross_media_recommendation(self, item_id: int, top_k: int = 5) -> list:
        """
        Given a Book ID -> returns matching Movies (and vice-versa) based on semantic tone.
        """
        target_idx_series = self.df.index[self.df["id"] == item_id]
        if target_idx_series.empty:
            return []

        target_idx = target_idx_series[0]
        source_item = self.df.iloc[target_idx].to_dict()
        source_type = source_item.get("type", "book")
        target_type = "movie" if source_type == "book" else "book"

        item_vec = self.embeddings[target_idx].reshape(1, -1)
        similarities = cosine_similarity(item_vec, self.embeddings)[0]

        ranked_indices = similarities.argsort()[::-1]
        results = []

        for idx in ranked_indices:
            if idx == target_idx:
                continue
            row = self.df.iloc[idx].to_dict()
            if row.get("type") == target_type:
                row["match_score"] = round(float(similarities[idx]) * 100, 1)
                results.append(row)

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    recommender = SemanticRecommender()
    print("\n--- Testing Vibe Search: 'cyberpunk virtual reality hacking' ---")
    matches = recommender.search_by_vibe("cyberpunk virtual reality hacking", top_k=3)
    for m in matches:
        print(f"[{m['type'].upper()}] {m['title']} - {m['match_score']}% Match")

    print("\n--- Testing Cross-Media Match for Dune (Book -> Movie) ---")
    cross = recommender.get_cross_media_recommendation(item_id=1, top_k=3)
    for c in cross:
        print(f"[{c['type'].upper()}] {c['title']} - {c['match_score']}% Match")
