# StoryBridge AI | Cross-Media Semantic Recommendation Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![NLP](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> A full-stack recommendation platform that bridges literature and cinema using applied Natural Language Processing (NLP). Discover movies matching the mood, theme, and philosophical depth of your favorite books (and vice-versa).

---

## 📌 Problem & Concept

Traditional recommendation engines operate in content silos: books recommend books, and movies recommend movies, relying on rigid keywords or basic genre tags (e.g., "Sci-Fi"). A reader captivated by the atmospheric dread and philosophical intrigue of Frank Herbert's *Dune* cannot easily find cinematic spiritual successors without manual research.

**StoryBridge AI** maps both literature and cinematic titles into a **shared high-dimensional semantic vector space**. Using transformer embeddings and cosine similarity, users can perform natural-language "vibe" queries or select any title to find cross-media matches with quantifiable similarity scores.

---

## 🏛️ Architecture Overview

```
                      ┌─────────────────────────────────────────┐
                      │            CLIENT (Frontend)            │
                      │  Responsive Netflix-Style UI (HTML/CSS) │
                      │  - Dynamic Search & Filter Tabs         │
                      │  - Cross-Media Matching Badges          │
                      └────────────────────┬────────────────────┘
                                           │ REST API (JSON)
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │          API LAYER (FastAPI)            │
                      │  - GET  /api/catalog                    │
                      │  - POST /api/recommend (Semantic Search)│
                      │  - GET  /api/cross-media/{id}           │
                      └────────────────────┬────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
      ┌──────────────────────────────────┐ ┌──────────────────────────────────┐
      │         CATALOG STORAGE          │ │        ML EMBEDDING ENGINE       │
      │        data/catalog.json         │ │   Sentence-Transformers (NLP)    │
      │ (54+ Curated Books & Movies)     │ │   - all-MiniLM-L6-v2             │
      │                                  │ │   - Vector Cosine Similarity     │
      └──────────────────────────────────┘ └──────────────────────────────────┘
```

---

## ✨ Key Features

1. **Natural-Language "Vibe" Search**: Search by atmosphere or narrative tropes (e.g., *"rainy neo-noir detective mystery"* or *"isolated space station survival"*).
2. **Cross-Media Matching ("Read This → Watch That")**: Click on any book to find films with identical thematic tone, or click on a movie to discover related literature.
3. **Similarity Score Badges**: Displays percentage match indicators (e.g., `94% Match`) calculated from vector cosine similarity.
4. **Curated Multi-Genre Catalog**: 50+ pre-indexed titles across Sci-Fi, Dark Fantasy, Historical Drama, Cyberpunk, and Mystery.
5. **Decoupled Full-Stack Design**: Modern client interface with asynchronous API fallbacks for standalone or backend-connected operation.

---

## 📂 Project Structure

```
├── backend/
│   └── main.py              # FastAPI REST endpoints & CORS configuration
├── ml/
│   └── recommender.py       # Sentence-Transformers semantic search engine
├── data/
│   └── catalog.json         # Structured catalog of books, movies, and synopses
├── index.html               # Main user discovery interface
├── signup.html              # Authentication / Onboarding portal
├── requirements.txt         # Python dependencies
├── .gitignore               # Environment and cache ignore rules
└── README.md                # Documentation & Architecture
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/bhaktee12/StoryBridge-AI.git
cd StoryBridge-AI
```

### 2. Backend & ML Setup (Python)
Ensure Python 3.8+ is installed.

```bash
# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI service
uvicorn backend.main:app --reload --port 8000
```
* Interactive API documentation will be live at: **http://127.0.0.1:8000/docs**

### 3. Frontend
Simply open `index.html` in any modern web browser or serve it with Live Server:
```bash
# Optional: using Python's built-in HTTP server
python -m http.server 3000
```
Open **http://localhost:3000** in your browser.

---

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/catalog` | `GET` | Retrieve catalog items with optional `type` and `genre` filters |
| `/api/recommend` | `POST` | Semantic vector search across descriptions using natural-language prompt |
| `/api/cross-media/{id}` | `GET` | Find opposing media titles with high semantic similarity |


