#!/usr/bin/env python3
"""
RAG chat script with robust Qdrant compatibility and targeted SQL fallback for job-title queries.

- Tries Qdrant query_points -> search -> REST.
- If vector retrieval is insufficient for job-title queries (e.g. "details about senior network analyst"),
  runs a safe SQL fallback across Requirements, RecruitmentRoles, Candidates, ApplicationResumes for JobTitle/Role/Skills/resume text.
- Converts SQL rows to docs and feeds them to the LLM; LLM answers using only those records (and cites sources).
- Environment variables required: QDRANT_URL, QDRANT_API_KEY. Optional ones: QDRANT_COLLECTION, MSSQL_CONNECTION, HF_EMBEDDING_MODEL, LLM_MODEL.
"""

import os
import re
import traceback
from typing import List, Dict, Optional, Any

import sqlalchemy as sa
from sqlalchemy import inspect, text as sa_text
from sqlalchemy.engine import Engine

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

# embeddings: try recommended package first
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

# http client fallback for REST qdrant
try:
    import httpx as _httpx
    _HAS_HTTPX = True
except Exception:
    import requests as _requests
    _HAS_HTTPX = False

# ----------------------------
# Config
# ----------------------------
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mssql_jobpost_vectors")
CONNECTION_STRING = os.getenv(
    "MSSQL_CONNECTION",
    "mssql+pyodbc://sa:Strong!12345@66.179.82.107,1433/Jobpost?driver=ODBC+Driver+17+for+SQL+Server"
)
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
MODEL_NAME = os.getenv("LLM_MODEL", "gemini-3.0-generate-001")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
TOP_K = int(os.getenv("TOP_K", "50"))
COUNT_K = int(os.getenv("COUNT_K", "100"))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in environment.")

# ----------------------------
# Globals & singletons
# ----------------------------
engine: Engine = sa.create_engine(CONNECTION_STRING, pool_pre_ping=True)
TABLE_SCHEMAS: Dict[str, Dict] = {}
TABLE_COUNTS: Dict[str, int] = {}

_qdrant_client: Optional[QdrantClient] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None
_payload_index_ready: Optional[bool] = None

# ----------------------------
# Qdrant helpers (connect + search with fallbacks)
# ----------------------------
def connect_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
    return _qdrant_client

def qdrant_http_search(query_vector: List[float], k: int, payload_filter: Optional[dict]):
    url = QDRANT_URL.rstrip("/") + f"/collections/{QDRANT_COLLECTION}/points/search"
    headers = {"Content-Type": "application/json", "api-key": QDRANT_API_KEY}
    body = {
        "vector": query_vector,
        "limit": k,
        "with_payload": True,
        "with_vectors": False
    }
    if payload_filter:
        body["filter"] = payload_filter
    if _HAS_HTTPX:
        resp = _httpx.post(url, headers=headers, json=body, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
    else:
        resp = _requests.post(url, headers=headers, json=body, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
    return data.get("result", data.get("points", []))

def qdrant_search(client: QdrantClient, query_vector: List[float], k: int, payload_filter: Optional[dict]):
    # Try client.query_points, then client.search, then HTTP REST
    try:
        kwargs = dict(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False
        )
        if payload_filter:
            kwargs["filter"] = payload_filter
        return client.query_points(**kwargs)
    except Exception as e_qp:
        try:
            kwargs = dict(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
                with_vectors=False
            )
            if payload_filter:
                kwargs["query_filter"] = payload_filter
            return client.search(**kwargs)
        except UnexpectedResponse:
            raise
        except Exception:
            pass
    return qdrant_http_search(query_vector, k, payload_filter)

def normalize_qdrant_point(pt: Any) -> Dict[str, Any]:
    payload = {}
    score = None
    pid = None
    if hasattr(pt, "payload"):
        payload = pt.payload or {}
    elif isinstance(pt, dict):
        payload = pt.get("payload", {}) or {}
        score = pt.get("score") or pt.get("dist")
        pid = pt.get("id")
    if hasattr(pt, "score") and score is None:
        score = getattr(pt, "score")
    if hasattr(pt, "id") and pid is None:
        pid = getattr(pt, "id")
    return {"payload": payload, "score": score, "id": pid}

def extract_text_from_payload(payload: Dict[str, Any]) -> str:
    candidates = ["text", "content", "page_content", "body", "resume_text", "description", "summary", "notes", "title", "name", "skills"]
    parts = []
    for key in candidates:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, list):
            parts.append(", ".join(str(x) for x in val if isinstance(x, (str, int, float))))
    # other small textual fields
    for k, v in payload.items():
        if k in candidates:
            continue
        if isinstance(v, str) and len(v) < 400:
            parts.append(f"{k}: {v}")
    if not parts:
        short_items = []
        for k, v in payload.items():
            if isinstance(v, (str, int, float)) and len(str(v)) < 200:
                short_items.append(f"{k}: {v}")
            if len(short_items) >= 6:
                break
        parts = short_items
    return " | ".join(parts)

# ----------------------------
# Schema cache
# ----------------------------
def build_schema_cache():
    global TABLE_SCHEMAS, TABLE_COUNTS
    if TABLE_SCHEMAS:
        return
    with engine.connect() as conn:
        inspector = inspect(conn)
        try:
            tables = inspector.get_table_names()
        except Exception:
            tables = []
        for table in tables:
            try:
                cols = [c["name"] for c in inspector.get_columns(table)]
                pk_info = inspector.get_pk_constraint(table)
                pk = pk_info.get("constrained_columns", [None])[0]
                count = conn.execute(sa_text(f"SELECT COUNT(*) FROM [{table}]")).fetchone()[0]
                TABLE_SCHEMAS[table] = {"columns": cols, "pk": pk}
                TABLE_COUNTS[table] = count
            except Exception:
                continue

# ----------------------------
# Router (fast + LLM fallback)
# ----------------------------
def route_query_fast(query: str, threshold: int = 1) -> List[str]:
    q = query.lower()
    tokens = set(re.findall(r"\w+", q))
    scores: Dict[str, int] = {}
    for table, meta in TABLE_SCHEMAS.items():
        score = 0
        table_tokens = set(re.findall(r"\w+", table.lower()))
        score += len(tokens & table_tokens) * 2
        for col in meta.get("columns", []):
            col_tokens = set(re.findall(r"\w+", col.lower()))
            if tokens & col_tokens:
                score += 1
        if score > 0:
            scores[table] = score
    if scores:
        top = [t for t, s in sorted(scores.items(), key=lambda kv: -kv[1])]
        return top[:3]
    return []

def route_query_with_llm(llm: ChatGoogleGenerativeAI, query: str) -> List[str]:
    fast = route_query_fast(query)
    if fast:
        return fast
    router_prompt = f"""You are a SQL table router. Choose the most relevant tables from this list:
{list(TABLE_SCHEMAS.keys())}
User query: "{query}"
Return a Python list like ["Users"] and only valid table names.
"""
    try:
        out = llm.invoke(router_prompt, temperature=0.0).content.strip()
        t_list = eval(out) if out else []
        if isinstance(t_list, list):
            return [t for t in t_list if t in TABLE_SCHEMAS][:3]
    except Exception:
        pass
    return []

# ----------------------------
# Job-title detection and SQL fallback
# ----------------------------
def extract_job_title_from_query(query: str) -> Optional[str]:
    q = query.strip()
    # common patterns: "details about X", "details on X", "tell me about X", "about X"
    patterns = [
        r"(?:details about|details on|tell me about|information about|about|show me details about)\s+(['\"]?)(?P<title>[^'\"]+?)\1\s*$",
        r"^what (?:is|are) the .* (?:for|about)\s+(?P<title>.+)$",
        r"^find jobs for (?P<title>.+)$",
    ]
    for patt in patterns:
        m = re.search(patt, q, flags=re.IGNORECASE)
        if m:
            title = m.groupdict().get("title", "").strip()
            # trim trailing question mark or punctuation
            title = re.sub(r"[?\.!]+$", "", title).strip()
            if title:
                return title
    # fallback: if query begins with a noun phrase like "senior network analyst" alone
    if len(q.split()) <= 5 and " " in q:
        # treat as possible job title if contains senior/junior/analyst/engineer/manager keywords
        if re.search(r"\b(senior|jr|junior|lead|principal|analyst|engineer|developer|architect|manager)\b", q, flags=re.IGNORECASE):
            return q
    return None

def sql_find_by_job_title(title: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Search several tables for job titles / roles matching the title using LIKE on relevant columns.
    Returns list of rows (dict) with a "table" key added.
    """
    if not title:
        return []
    t = title.lower()
    like = f"%{t}%"
    results = []
    queries = [
        ("Requirements", "SELECT TOP :limit * FROM [Requirements] WHERE LOWER(JobTitle) LIKE :like OR LOWER(Skills) LIKE :like"),
        ("RecruitmentRoles", "SELECT TOP :limit * FROM [RecruitmentRoles] WHERE LOWER(Role) LIKE :like"),
        ("Candidates", "SELECT TOP :limit * FROM [Candidates] WHERE LOWER(CurrentJobTitle) LIKE :like OR LOWER(Title) LIKE :like"),
        ("ApplicationResumes", "SELECT TOP :limit * FROM [ApplicationResumes] WHERE LOWER(CONCAT(ISNULL(ResumeText, ''), ' ', ISNULL(Summary, ''))) LIKE :like"),
    ]
    try:
        with engine.connect() as conn:
            for table_name, sql in queries:
                try:
                    rows = conn.execute(sa_text(sql), {"limit": limit, "like": like}).mappings().all()
                    for r in rows:
                        d = dict(r)
                        d["table"] = table_name
                        results.append(d)
                except Exception:
                    # ignore per-table errors and continue
                    continue
    except Exception:
        return []
    return results

# ----------------------------
# Embeddings & helper
# ----------------------------
def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    return _embeddings

def embed_query_vector(query: str) -> List[float]:
    emb = get_embeddings()
    try:
        vec = emb.embed_query(query)
    except Exception:
        vecs = emb.embed_documents([query])
        vec = vecs[0]
    try:
        return list(vec)
    except Exception:
        return vec

# ----------------------------
# High-level retrieval: vector -> docs (then SQL fallback for job-title)
# ----------------------------
def build_payload_filter(tables: List[str]) -> Optional[dict]:
    if not tables:
        return None
    must = [{"key": tkey, "match": {"value": t}} for t in []]  # placeholder; not used
    # We keep simpler shape used earlier:
    must = []
    for t in tables:
        must.append({"key": "table", "match": {"value": t}})
    return {"must": must}

def retrieve_documents(client: QdrantClient, query: str, tables: List[str], k: int, payload_index_available: bool) -> List[Any]:
    vec = embed_query_vector(query)
    payload_filter = build_payload_filter(tables) if (tables and payload_index_available) else None

    try:
        pts = qdrant_search(client, vec, k, payload_filter)
    except UnexpectedResponse as e:
        msg = str(e)
        if "Index required" in msg or "index required" in msg.lower():
            print("⚠️ Qdrant payload index missing for the filter key. Retrying without payload filter.")
            pts = qdrant_search(client, vec, k, None)
        else:
            raise

    docs = []
    for pt in pts:
        norm = normalize_qdrant_point(pt)
        payload = norm["payload"] or {}
        text = extract_text_from_payload(payload)
        doc = type("Doc", (), {})()
        doc.metadata = payload
        doc.page_content = text
        docs.append(doc)

    # If not enough vector docs and query is a job-title request, run SQL fallback
    job_title = extract_job_title_from_query(query)
    if job_title and (len(docs) == 0 or all((not getattr(d, "page_content", "")) for d in docs)):
        rows = sql_find_by_job_title(job_title, limit=k)
        for r in rows:
            doc = type("Doc", (), {})()
            payload = dict(r)
            # Build page content prioritizing JobTitle/Role/Skills/resume text
            parts = []
            for key in ("JobTitle", "Role", "CurrentJobTitle", "Skills", "ResumeText", "Summary"):
                if key in payload and payload[key]:
                    parts.append(f"{key}: {payload[key]}")
            # add few metadata fields
            for key in ("Location", "Experience", "Company"):
                if key in payload and payload[key]:
                    parts.append(f"{key}: {payload[key]}")
            doc.metadata = payload
            doc.page_content = " | ".join(parts) or str(payload)
            docs.insert(0, doc)
    return docs

# ----------------------------
# Prompting and chat loop
# ----------------------------
def chat():
    print("Starting RAG chat (job-title SQL fallback enabled)...\n")
    build_schema_cache()

    client = connect_qdrant()
    try:
        client.get_collection(QDRANT_COLLECTION)
    except Exception as e:
        print("Qdrant collection error:", e)
        return

    # Try to ensure payload index exists; code omitted here for brevity (keep prior implementation if needed)
    # payload_index_available = ensure_table_payload_index(client)
    # For safety assume False -> we'll still use client-side fallback where necessary
    payload_index_available = False

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE, convert_system_message_to_human=True)
    print("✅ Ready. Type messages (type 'exit' or 'quit' to stop).\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower().strip() in ("quit", "exit"):
            break

        tables = route_query_with_llm(llm, q)
        print("Router:", tables or "global")

        is_count = any(k in q.lower() for k in ["count", "total", "how many"])
        k = COUNT_K if is_count else TOP_K
        if len(tables) > 2:
            k = max(20, int(k / len(tables)))

        try:
            docs = retrieve_documents(client, q, tables, k, payload_index_available)
        except Exception as e:
            print("Retrieval error:", type(e).__name__, e)
            traceback.print_exc(limit=1)
            docs = []

        # prioritize routed tables
        if tables:
            docs = sorted(docs, key=lambda d: 0 if d.metadata.get("table") in tables or d.metadata.get("source") == "SQL_latest_row" else 1)[:k]
        else:
            docs = docs[:k]

        table_summary = "\n".join(f"{t} ({TABLE_COUNTS.get(t,0)} rows) columns: {TABLE_SCHEMAS[t]['columns'][:8]}" for t in tables) if tables else "No specific table selected."
        count_summary = "\n".join(f"FACT: {t} has exactly {TABLE_COUNTS.get(t,0)} records." for t in tables) if tables else ""
        record_lines = []
        for d in docs:
            src_table = d.metadata.get("table") or d.metadata.get("source") or "unknown"
            snippet = (d.page_content[:800] + "...") if len(d.page_content) > 800 else d.page_content
            record_lines.append(f"[{src_table}] {snippet}")
        record_text = "\n".join(record_lines) or "No retrieved records."

        final_prompt = f"""You are an analytical assistant. Use ONLY the records below and do not hallucinate.

=== TABLE INFO ===
{table_summary}

{count_summary}

=== RECORDS ===
{record_text}

User question: {q}

Rules:
- Answer using only RECORDS. Cite sources like [table_name].
- If you can fully answer, be concise. If partial, say what extra info is needed.
- If nothing useful, reply "Insufficient data" with a suggested next action.
"""
        try:
            response = llm.invoke(final_prompt)
            out = getattr(response, "content", None)
            print("\nAI:", (out or str(response)).strip(), "\n")
        except Exception as e:
            print("LLM error:", type(e).__name__, e)
            traceback.print_exc(limit=1)
            print("\nAI: (LLM error) Please try again later.\n")

if __name__ == "__main__":
    chat()