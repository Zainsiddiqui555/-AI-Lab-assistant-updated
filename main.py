import os
import uuid
import json
import logging
from typing import Optional, List, Dict
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lab-assistant")

# ─── Configuration ───────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "lab_documents")
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set. AI features will fail.")

# ─── Clients ─────────────────────────────────────────────────────────────────

openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=ChromaSettings(anonymized_telemetry=False),
)

# In-memory webhook registry (replace with DB in production)
webhook_registry: Dict[str, dict] = {}

# ─── App Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Lab Assistant starting up")
    _ensure_collection()
    yield
    logger.info("AI Lab Assistant shutting down")

app = FastAPI(
    title="AI Lab Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Models ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The lab question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    chunks_retrieved: int

class WebhookRegister(BaseModel):
    url: str = Field(..., description="Callback URL")
    secret: Optional[str] = None
    label: Optional[str] = None

class WebhookIngestPayload(BaseModel):
    text: str = Field(..., description="Lab data text to index")
    source: Optional[str] = Field(default="webhook", description="Source label")
    metadata: Optional[Dict[str, str]] = None

class WebhookQueryPayload(BaseModel):
    question: str
    top_k: Optional[int] = None
    callback_url: Optional[str] = None

class UploadResponse(BaseModel):
    status: str
    doc_id: str
    filename: str
    pages: int
    chunks: int
    characters: int

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    pages: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_indexed: int
    embedding_model: str
    llm_model: str
    webhooks_registered: int

# ─── ChromaDB Helpers ────────────────────────────────────────────────────────

def _ensure_collection():
    try:
        return chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        return chroma_client.create_collection(COLLECTION_NAME)

def get_collection():
    return _ensure_collection()

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)

# ─── Embedding ───────────────────────────────────────────────────────────────

def get_embedding(text: str) -> List[float]:
    if not OPENROUTER_API_KEY:
        raise HTTPException(503, "OpenRouter API key not configured")
    try:
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(502, f"Embedding service error: {str(e)}")

# ─── Text Chunking ───────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    chunk_size = chunk_size or MAX_CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "! ", "? ", "; ", ", "]:
                idx = text.rfind(sep, start, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ─── PDF Processing ──────────────────────────────────────────────────────────

async def process_pdf(file: UploadFile) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    try:
        pdf = PdfReader(content)
    except Exception as e:
        raise HTTPException(400, f"Invalid PDF: {str(e)}")

    pages_text = []
    for page in pdf.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise HTTPException(400, "No extractable text found in PDF")

    doc_id = str(uuid.uuid4())
    filename = file.filename

    chunks = chunk_text(full_text)

    collection = get_collection()
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        for j, chunk in enumerate(batch):
            idx = i + j
            chunk_id = f"{doc_id}_chunk_{idx}"
            emb = get_embedding(chunk)
            ids.append(chunk_id)
            embeddings.append(emb)
            metadatas.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "source": "pdf",
            })
            documents.append(chunk)
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    logger.info(f"Indexed '{filename}': {len(chunks)} chunks, {len(pdf.pages)} pages, {len(full_text)} chars")

    return {
        "doc_id": doc_id,
        "filename": filename,
        "pages": len(pdf.pages),
        "chunks": len(chunks),
        "characters": len(full_text),
    }

# ─── RAG Query ───────────────────────────────────────────────────────────────

def query_rag(question: str, top_k: int = None) -> dict:
    top_k = top_k or TOP_K
    collection = get_collection()

    count = collection.count()
    if count == 0:
        return {
            "answer": "No documents have been uploaded yet. Please upload PDF files first.",
            "sources": [],
            "chunks_retrieved": 0,
        }

    q_emb = get_embedding(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, count),
        include=["documents", "metadatas"],
    )

    if not results["documents"] or not results["documents"][0]:
        return {
            "answer": "No relevant content found for your question.",
            "sources": [],
            "chunks_retrieved": 0,
        }

    contexts = []
    source_set = {}
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        contexts.append(doc.strip())
        if meta and meta.get("filename"):
            fn = meta["filename"]
            if fn not in source_set:
                source_set[fn] = 0
            source_set[fn] += 1

    context_block = "\n\n---\n\n".join(contexts)
    sources = sorted(source_set.keys())

    system_prompt = (
        "You are an expert AI Lab Assistant. Answer questions based solely on the provided context. "
        "If the context lacks sufficient information, state what is missing. "
        "Cite source filenames when referencing specific documents. "
        "For numerical data, present it in tables when helpful. "
        "Be precise, scientific, and clear. Format responses in Markdown."
    )

    user_prompt = f"""Context from laboratory documents:
{context_block}

Question: {question}

Answer based on the context above. Include relevant citations from sources: {', '.join(sources)}"""

    if not OPENROUTER_API_KEY:
        raise HTTPException(503, "OpenRouter API key not configured")

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise HTTPException(502, f"LLM service error: {str(e)}")

    return {
        "answer": answer,
        "sources": sources,
        "chunks_retrieved": len(contexts),
    }

# ─── Webhook Processing ──────────────────────────────────────────────────────

async def send_webhook_callback(url: str, data: dict, secret: Optional[str] = None):
    import httpx
    try:
        headers = {"Content-Type": "application/json"}
        if secret:
            headers["X-Webhook-Secret"] = secret
        async with httpx.AsyncClient(timeout=30.0) as hc:
            resp = await hc.post(url, json=data, headers=headers)
            resp.raise_for_status()
        logger.info(f"Webhook callback sent to {url}")
    except Exception as e:
        logger.error(f"Webhook callback failed to {url}: {e}")

async def process_webhook_ingest(payload: WebhookIngestPayload, hook_id: str):
    try:
        text = payload.text.strip()
        if not text:
            return

        doc_id = f"webhook_{uuid.uuid4().hex[:12]}"
        source = payload.source or "webhook"
        meta = payload.metadata or {}

        chunks = chunk_text(text)
        collection = get_collection()

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            emb = get_embedding(chunk)
            collection.add(
                ids=[chunk_id],
                embeddings=[emb],
                metadatas=[{
                    "doc_id": doc_id,
                    "filename": f"{source}_{doc_id}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": "webhook",
                    "webhook_id": hook_id,
                    **meta,
                }],
                documents=[chunk],
            )

        logger.info(f"Webhook ingest: {len(chunks)} chunks from '{source}'")
    except Exception as e:
        logger.error(f"Webhook ingest failed: {e}")

async def process_webhook_query(payload: WebhookQueryPayload, hook_id: str):
    try:
        result = query_rag(payload.question, payload.top_k)
        callback_url = payload.callback_url
        if not callback_url and hook_id in webhook_registry:
            callback_url = webhook_registry[hook_id].get("url")

        if callback_url:
            await send_webhook_callback(
                callback_url,
                {"question": payload.question, **result},
                webhook_registry.get(hook_id, {}).get("secret"),
            )
    except Exception as e:
        logger.error(f"Webhook query processing failed: {e}")

# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AI Lab Assistant</h1><p>Frontend not found. Ensure index.html is in the same directory as main.py</p>")

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    result = await process_pdf(file)
    return {"status": "success", **result}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(422, "Question cannot be empty")
    result = query_rag(req.question, req.top_k)
    return {"question": req.question, **result}

# ─── Webhook Endpoints ───────────────────────────────────────────────────────

@app.post("/webhook/register")
async def register_webhook(payload: WebhookRegister):
    hook_id = str(uuid.uuid4())
    webhook_registry[hook_id] = {
        "url": payload.url,
        "secret": payload.secret,
        "label": payload.label or "unnamed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(f"Webhook registered: {hook_id} -> {payload.url}")
    return {"status": "registered", "webhook_id": hook_id}

@app.get("/webhook/registry")
async def list_webhooks():
    return {
        "webhooks": [
            {"id": hid, **info}
            for hid, info in webhook_registry.items()
        ]
    }

@app.delete("/webhook/registry/{hook_id}")
async def delete_webhook(hook_id: str):
    if hook_id in webhook_registry:
        del webhook_registry[hook_id]
        return {"status": "deleted"}
    raise HTTPException(404, "Webhook not found")

@app.post("/webhook/ingest")
async def webhook_ingest(payload: WebhookIngestPayload, background_tasks: BackgroundTasks):
    hook_id = "anonymous"
    background_tasks.add_task(process_webhook_ingest, payload, hook_id)
    return {"status": "accepted", "message": "Data queued for indexing"}

@app.post("/webhook/query")
async def webhook_query(payload: WebhookQueryPayload, background_tasks: BackgroundTasks):
    hook_id = "anonymous"
    background_tasks.add_task(process_webhook_query, payload, hook_id)
    return {"status": "accepted", "message": "Query queued for processing"}

# ─── Document Management ─────────────────────────────────────────────────────

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    collection = get_collection()
    if collection.count() == 0:
        return []

    all_data = collection.get(include=["metadatas"])
    docs_map: Dict[str, dict] = {}
    for meta in all_data["metadatas"]:
        if not meta:
            continue
        did = meta.get("doc_id")
        if not did:
            continue
        if did not in docs_map:
            docs_map[did] = {
                "doc_id": did,
                "filename": meta.get("filename", "unknown"),
                "chunks": 0,
            }
        docs_map[did]["chunks"] += 1

    return list(docs_map.values())

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    collection = get_collection()
    all_data = collection.get(include=["metadatas"])
    to_delete = [
        all_data["ids"][i]
        for i, meta in enumerate(all_data["metadatas"])
        if meta and meta.get("doc_id") == doc_id
    ]
    if not to_delete:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    collection.delete(ids=to_delete)
    return {"status": "deleted", "chunks_removed": len(to_delete)}

@app.delete("/documents")
async def delete_all_documents():
    collection = get_collection()
    count = collection.count()
    if count > 0:
        all_ids = collection.get()["ids"]
        collection.delete(ids=all_ids)
    return {"status": "deleted", "total_chunks_removed": count}

# ─── System ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    collection = get_collection()
    count = collection.count()
    chunk_ids = collection.get()["ids"] if count > 0 else []
    unique_docs = len(set(
        m.get("doc_id") for m in collection.get(include=["metadatas"])["metadatas"] if m
    )) if count > 0 else 0

    return HealthResponse(
        status="healthy",
        documents_indexed=unique_docs,
        chunks_indexed=count,
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        webhooks_registered=len(webhook_registry),
    )

# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
