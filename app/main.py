"""FastAPI entry: health check and stateless chat endpoint."""
from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

# Load project-root .env before agent reads GROQ_API_KEY / OPENROUTER_API_KEY
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agent import run_agent
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retriever import HybridRetriever
from app.utils import setup_logging

LOG = logging.getLogger("shl_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    retriever = HybridRetriever()
    retriever.load()
    app.state.retriever = retriever
    LOG.info("Retriever ready (%d items)", len(retriever._items))
    yield


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()
    retriever: HybridRetriever = request.app.state.retriever
    resp = run_agent(body.messages, retriever)
    ms = (time.perf_counter() - t0) * 1000
    LOG.info(
        "chat_done id=%s ms=%.1f recs=%d eoc=%s",
        rid,
        ms,
        len(resp.recommendations),
        resp.end_of_conversation,
    )
    return resp

@app.get("/")
def root():
    return {
        "message": "SHL Conversational Recommender API is running"
    }
