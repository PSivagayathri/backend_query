
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Depends, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import requests
import fitz  # PyMuPDF
from docx import Document
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Client
import json
import re
import time
import psycopg2
from psycopg2.extras import Json
import magic  # pip install python-magic
from dotenv import load_dotenv
import os

# =======================
# Load environment variables
# =======================
load_dotenv()  # Load from .env file

API_PREFIX = "/api/v1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))

# =======================
# FastAPI Init
# =======================
app = FastAPI(title="HackRx Style Document Q&A API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix=API_PREFIX)

# =======================
# Auth Dependency
# =======================
def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {BEARER_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

# =======================
# Global State
# =======================
model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = []
embeddings = None
index = None
groq_client = Client(api_key=GROQ_API_KEY)

# =======================
# PostgreSQL Connection
# =======================
conn = psycopg2.connect(
    host=POSTGRES_HOST,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    port=POSTGRES_PORT
)
conn.autocommit = True

def log_to_db(question, answer, conditions, rationale, execution_time):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO query_logs (question, answer, conditions, rationale, execution_time)
            VALUES (%s, %s, %s, %s, %s)
        """, (question, answer, Json(conditions), rationale, execution_time))

# =======================
# Utils
# =======================
def parse_pdf(file_path):
    doc = fitz.open(file_path)
    return " ".join([page.get_text() for page in doc])

def parse_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def chunk_text(text, max_tokens=300):
    sentences = text.split('. ')
    chunk_list, current, count = [], [], 0
    for s in sentences:
        words = len(s.split())
        if count + words > max_tokens:
            chunk_list.append(". ".join(current))
            current = [s]
            count = words
        else:
            current.append(s)
            count += words
    if current:
        chunk_list.append(". ".join(current))
    return chunk_list

def build_index(chunks_list):
    global embeddings, index
    embeddings = model.encode(chunks_list)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

def search(query, top_k=3):
    query_vec = model.encode([query])
    _, indices = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in indices[0]]

def query_llm(query, context):
    prompt = f"""
You are an intelligent document analysis agent. Answer the user's question based on the context.

Context:
{context}

Question: {query}

Respond strictly in JSON:
{{
  "answer": "<short answer>",
  "conditions": ["<condition 1>", "<condition 2>"],
  "rationale": "<explanation>"
}}
"""
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

def safe_parse_llm_response(raw_output: str) -> dict:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e2:
                return {"error": f"Could not parse extracted JSON: {str(e2)}"}
        return {"error": "Failed to parse LLM response"}

# =======================
# Models
# =======================
class MultiQueryRequest(BaseModel):
    queries: List[str]

class HackRxRunRequest(BaseModel):
    documents: str
    questions: List[str]

# =======================
# API Endpoints
# =======================
@router.post("/upload", dependencies=[Depends(verify_token)])
def upload_doc(file: UploadFile = File(...)):
    global chunks

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        file_path = tmp.name

    mime_type = magic.from_file(file_path, mime=True)
    print("Detected MIME type:", mime_type)

    if mime_type == "application/pdf":
        text = parse_pdf(file_path)
    elif mime_type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        text = parse_docx(file_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")

    chunks = chunk_text(text)
    build_index(chunks)
    return {"status": "Document parsed and indexed", "chunks": len(chunks)}

@router.post("/ask", dependencies=[Depends(verify_token)])
def ask_multiple(request: MultiQueryRequest):
    if not index:
        return [{"error": "No document uploaded yet."} for _ in request.queries]

    results = []
    for query in request.queries:
        start_time = time.time()
        context = "\n".join(search(query))
        raw = query_llm(query, context)
        exec_time = time.time() - start_time

        parsed = safe_parse_llm_response(raw)

        if "error" not in parsed:
            log_to_db(
                question=query,
                answer=parsed.get("answer", ""),
                conditions=parsed.get("conditions", []),
                rationale=parsed.get("rationale", ""),
                execution_time=exec_time
            )

        results.append(parsed)
    return results

@router.post("/hackrx/run", dependencies=[Depends(verify_token)])
def hackrx_run(req: HackRxRunRequest):
    global chunks

    # Download file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        r = requests.get(req.documents)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")
        tmp.write(r.content)
        file_path = tmp.name

    # Detect type
    mime_type = magic.from_file(file_path, mime=True)
    print("Detected MIME type:", mime_type)

    if mime_type == "application/pdf":
        text = parse_pdf(file_path)
    elif mime_type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        text = parse_docx(file_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")

    chunks = chunk_text(text)
    build_index(chunks)

    results = []
    for question in req.questions:
        start_time = time.time()
        context = "\n".join(search(question))
        raw = query_llm(question, context)
        exec_time = time.time() - start_time

        parsed = safe_parse_llm_response(raw)
        if "error" not in parsed:
            log_to_db(question, parsed.get("answer", ""), parsed.get("conditions", []),
                      parsed.get("rationale", ""), exec_time)
        results.append(parsed)

    return results

@router.get("/health")
def health():
    return {"status": "OK"}

# Register router
app.include_router(router)






