import os
import textwrap
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import inspect, text as sa_text
from sqlalchemy.engine import Engine

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# ============================
# CONFIG
# ============================

QDRANT_COLLECTION = "sql_rag_vectors"

CONNECTION_STRING = (
    "mssql+pyodbc://sa:Strong!12345@66.179.82.107,1433/Jobpost?"
    "driver=ODBC+Driver+17+for+SQL+Server"
)

HF_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.7

TOP_K = 50
COUNT_K = 100


# ============================
# GLOBALS
# ============================

engine: Engine = sa.create_engine(CONNECTION_STRING, pool_pre_ping=True)
TABLE_SCHEMAS: Dict[str, Dict] = {}
TABLE_COUNTS: Dict[str, int] = {}
vector_db = None
retriever = None
llm = None


# ============================
# QDRANT
# ============================

def connect_qdrant():
    return QdrantClient(
        url="https://fc3322b9-00ce-428a-ad41-84067d5046de.eu-central-1-0.aws.cloud.qdrant.io",
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=60.0
    )


# ============================
# SCHEMA CATCHER
# ============================

def powerful_schema_catcher(engine):
    schema_meta = {}

    with engine.connect() as conn:
        inspector = inspect(conn)
        tables = inspector.get_table_names()

        for table in tables:
            try:
                columns = [c["name"] for c in inspector.get_columns(table)]

                try:
                    pk_info = inspector.get_pk_constraint(table)
                    pk = pk_info.get("constrained_columns", [None])[0]
                except:
                    pk = None

                try:
                    count = conn.execute(sa_text(f"SELECT COUNT(*) FROM [{table}]")).fetchone()[0]
                except:
                    count = 0

                safe_cols = []
                result_cols = inspector.get_columns(table)

                for col in result_cols:
                    dtype = str(col["type"]).lower()
                    if any(x in dtype for x in ["binary", "varbinary", "image", "xml", "geography", "hierarchyid"]):
                        continue
                    safe_cols.append(col["name"])

                sample_rows = []
                if safe_cols:
                    try:
                        q = f"SELECT TOP 3 {', '.join(safe_cols)} FROM [{table}]"
                        rows = conn.execute(sa_text(q)).fetchall()
                        sample_rows = [str(dict(r)) for r in rows]
                    except:
                        pass

                schema_meta[table] = {
                    "table": table,
                    "columns": columns,
                    "pk": pk,
                    "count": count,
                    "sample_rows": sample_rows
                }

            except:
                continue

    return schema_meta


# ============================
# QDRANT INDEX BUILD
# ============================

def build_qdrant_index(schema_meta):
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    client = connect_qdrant()

    try:
        client.get_collection(QDRANT_COLLECTION)
    except:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    chunks = []
    metadatas = []

    for table, meta in schema_meta.items():
        txt = f"""
TABLE: {table}
COLUMNS: {meta['columns']}
PRIMARY KEY: {meta['pk']}
ROW COUNT: {meta['count']}
SAMPLE ROWS:
{meta['sample_rows']}
"""
        chunks.append(txt)
        metadatas.append({"table": table})

    Qdrant.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        client=client,
        collection_name=QDRANT_COLLECTION
    )


# ============================
# LOAD VECTOR STORE
# ============================

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    client = connect_qdrant()
    return Qdrant(client=client, collection_name=QDRANT_COLLECTION, embeddings=embeddings)


# ============================
# ROUTER
# ============================

def route_query(llm, query):
    tables = list(TABLE_SCHEMAS.keys())

    router_prompt = f"""
You are an SQL table router.
Choose best matching tables from:

{tables}

User query: "{query}"

Return Python list only, e.g. ["Users"]
"""

    out = llm.invoke(router_prompt).content.strip()
    try:
        lst = eval(out)
        if isinstance(lst, list):
            return [t for t in lst if t in TABLE_SCHEMAS]
    except:
        pass

    return []


# ============================
# FASTAPI
# ============================

app = FastAPI(root_path="/chatbot")



class Question(BaseModel):
    question: str


@app.post("/chat")
def chat_endpoint(body: Question):
    q = body.question.strip()

    is_count = any(x in q.lower() for x in ["count", "total", "how many"])

    tables = route_query(llm, q)
    k = COUNT_K if is_count else TOP_K
    retriever.search_kwargs["k"] = k

    docs = retriever.invoke(q)

    docs = sorted(
        docs,
        key=lambda d: 0 if d.metadata.get("table") in tables else 1
    )[:k]

    table_summary = "\n".join(
        f"{t} ({TABLE_COUNTS[t]} rows) columns: {TABLE_SCHEMAS[t]['columns'][:8]}"
        for t in tables
    )

    count_summary = "\n".join(
        f"FACT: {t} = {TABLE_COUNTS[t]} records"
        for t in tables
    )

    record_text = "\n".join(
        f"[{d.metadata.get('table')}] {d.page_content}"
        for d in docs
    )

    final_prompt = f"""
You are an analytical SQL assistant.

=== TABLE INFO ===
{table_summary}

{count_summary}

=== RECORDS ===
{record_text}

User question: {q}

Rules:
- Answer ONLY using the information above.
- No guessing or hallucination.
"""

    try:
        answer = llm.invoke(final_prompt).content
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


# ============================
# STARTUP INITIALIZATION
# ============================

@app.on_event("startup")
def startup_event():
    global TABLE_SCHEMAS, TABLE_COUNTS, vector_db, retriever, llm

    TABLE_SCHEMAS = powerful_schema_catcher(engine)
    TABLE_COUNTS = {t: TABLE_SCHEMAS[t]["count"] for t in TABLE_SCHEMAS}

    client = connect_qdrant()
    try:
        client.get_collection(QDRANT_COLLECTION)
    except:
        build_qdrant_index(TABLE_SCHEMAS)

    vector_db = load_vector_store()
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE,google_api_key=os.getenv("GOOGLE_API_KEY"))


# ============================
# RUN
# ============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("RAG_REST:app", host="0.0.0.0", port=8000, reload=True)

