# SHL Conversational Assessment Recommender

A production-ready conversational AI system that recommends **only SHL Individual Test Solutions** through a stateless FastAPI API.

The system uses:

* Retrieval-Augmented Generation (RAG)
* FAISS semantic search
* Hybrid retrieval (semantic + keyword boosting)
* Groq-hosted LLMs
* Strict grounding and hallucination prevention

This project was built for the SHL AI Internship Assignment.

---

# Features

* Stateless conversational API
* Clarification handling for vague hiring requests
* Grounded SHL-only recommendations
* Multi-turn refinement support
* Assessment comparison support
* Prompt injection defense
* Off-topic refusal handling
* Hybrid semantic retrieval with FAISS
* Strict schema-compliant responses
* Docker + Render deployment ready

---

# Architecture Overview

```text
User Query
   ↓
FastAPI /chat endpoint
   ↓
Conversation routing (clarify / recommend / compare / refuse)
   ↓
Hybrid retrieval
   ├── FAISS semantic similarity
   └── Keyword/category boosting
   ↓
Grounded candidate shortlist
   ↓
LLM response generation (Groq/OpenRouter)
   ↓
Strict response validation
   ↓
JSON response
```

---

# Tech Stack

| Component     | Technology               |
| ------------- | ------------------------ |
| API           | FastAPI                  |
| Embeddings    | sentence-transformers    |
| Vector Search | FAISS                    |
| LLM           | Groq / OpenRouter        |
| Scraping      | BeautifulSoup + requests |
| Testing       | pytest                   |
| Deployment    | Render + Docker          |

---

# Project Structure

```text
project/
│
├── app/
│   ├── main.py
│   ├── agent.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── models.py
│   ├── utils.py
│   ├── catalog.json
│   └── faiss_index/
│
├── scraper/
│   └── scrape_shl.py
│
├── tests/
│   ├── test_chat.py
│   └── test_retrieval.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── approach_document.md
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone <your_repo_url>
cd shl-assessment-recommender
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

Optional:

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
LOG_LEVEL=INFO
```

If no LLM API key is configured, the API still returns grounded catalog matches using retrieval-only fallback mode.

---

# Build Catalog + FAISS Index

## Scrape SHL Catalog

```bash
python scraper/scrape_shl.py --out app/catalog.json
```

## Build Vector Index

```bash
python -m app.retriever --build
```

This:

* embeds SHL catalog entries
* builds FAISS index
* stores vector metadata under `app/faiss_index/`

---

# Run Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

---

# API Endpoints

## GET `/health`

Health check endpoint.

### Response

```json
{
  "status": "ok"
}
```

---

## POST `/chat`

Stateless conversational endpoint.

### Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "We are hiring mid-level Java developers."
    }
  ]
}
```

---

# Example Clarification Response

```json
{
  "reply": "Could you share the seniority level and whether you want technical, cognitive, or personality assessments?",
  "recommendations": [],
  "end_of_conversation": false
}
```

---

# Example Recommendation Response

```json
{
  "reply": "Here are recommended SHL assessments for mid-level Java backend developers.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": true
}
```

---

# Retrieval Pipeline

The recommender uses hybrid retrieval:

## 1. Semantic Search

* `sentence-transformers/all-MiniLM-L6-v2`
* FAISS cosine similarity search

## 2. Keyword & Category Boosting

Boosts:

* personality
* leadership
* OPQ
* behavioral
* technical skill keywords

Penalizes:

* generic reports
* framework-only entries
* documentation-style rows

This improves recommendation quality and Recall@10 performance.

---

# Safety & Grounding

The system includes:

* strict catalog grounding
* hallucination prevention
* URL allowlisting
* compare enforcement
* prompt injection defense
* off-topic refusal handling

The assistant never recommends assessments outside the SHL catalog.

---

# Running Tests

```bash
pytest tests -q
```

Current status:

```text
17 passed
```

---

# Docker

## Build

```bash
docker build -t shl-recommender .
```

## Run

```bash
docker run -p 8000:8000 -e GROQ_API_KEY=your_key shl-recommender
```

---

# Render Deployment

## Build Command

```bash
pip install -r requirements.txt
```

## Start Command

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Add environment variable in Render dashboard:

```env
GROQ_API_KEY=your_key
```

---

# Design Goals

This system was designed to:

* remain fully stateless
* prevent hallucinated recommendations
* support realistic hiring conversations
* optimize retrieval quality
* remain lightweight and interview-defensible
* satisfy SHL evaluation constraints

---

# Future Improvements

Potential future enhancements:

* better role-aware ranking
* metadata-aware filtering
* learning-to-rank retrieval
* stronger compare entity extraction
* evaluation trace replay tooling

---

# Author

Paras Dhajal


