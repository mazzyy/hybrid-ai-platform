#!/usr/bin/env python3
"""
E-T-A RAG Prototype - FastAPI REST API

Mirrors the production RAG API container.
Endpoints for querying, health checks, and admin operations.

Usage:
    python api/server.py
    # Then open http://localhost:8000/docs
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from retrieval.rag_engine import RAGEngine
from config.rbac import MOCK_USERS, ROLE_ACCESS
from config.settings import API_HOST, API_PORT

# ─── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="E-T-A RAG Prototype API",
    description="On-Premises Knowledge Retrieval for the Hybrid AI Platform (Prototype)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine (initialized on startup)
engine: Optional[RAGEngine] = None


# ─── Models ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    user_id: str = Field(default="david.braun", description="User ID for RBAC")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    user_role: str
    chunks_retrieved: int


class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
    llm_model: str
    embedding_model: str


# ─── Endpoints ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global engine
    engine = RAGEngine()


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        vector_store_count=engine.vector_store.count if engine else 0,
        llm_model=engine.model if engine else "not loaded",
        embedding_model="all-MiniLM-L6-v2",
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    The user_id determines RBAC access — different users see different results.
    Try: thomas.mueller (engineering), anna.weber (HR), david.braun (management)
    """
    if not engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    user = MOCK_USERS.get(request.user_id, {
        "name": "Unknown",
        "role": "all_staff",
        "department": "general",
    })

    result = engine.query(
        question=request.question,
        user_role=user["role"],
        user_name=user["name"],
        top_k=request.top_k,
    )

    return QueryResponse(**result)


@app.get("/users")
async def list_users():
    """List available mock users for RBAC testing."""
    return {
        uid: {
            "name": info["name"],
            "role": info["role"],
            "department": info["department"],
            "access_description": ROLE_ACCESS.get(info["role"], {}).get("description", ""),
        }
        for uid, info in MOCK_USERS.items()
    }


@app.get("/roles")
async def list_roles():
    """List RBAC role definitions."""
    return ROLE_ACCESS


if __name__ == "__main__":
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=True)
