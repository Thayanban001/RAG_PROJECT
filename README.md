# 🤖 RAG SQL Chatbot — Natural Language to Database

A production-grade AI chatbot that lets users query a **SQL database using plain English** — no SQL knowledge needed. Built with **FastAPI**, **LangChain**, **Qdrant**, and **Google Gemini**.

---

## 🧠 How It Works

```
User Question → Gemini Router → Qdrant Vector Search → Schema Context → Gemini LLM → Answer
```

1. User sends a natural language question via REST API
2. LLM **routes** the query to the relevant database tables
3. **Qdrant** vector store retrieves matching schema/sample data
4. **Gemini 2.5 Flash** generates an accurate, grounded answer
5. Response returned — no hallucination, no SQL required

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | Google Gemini 2.5 Flash |
| Vector Store | Qdrant Cloud |
| Embeddings | HuggingFace `all-mpnet-base-v2` |
| Orchestration | LangChain |
| Database | MS SQL Server (via SQLAlchemy) |
| Language | Python 3.10+ |

---

## 🚀 Features

- ✅ **Natural language querying** over any SQL database
- ✅ **Auto schema discovery** — reads all tables, columns, PKs, row counts & sample data
- ✅ **Smart table routing** — LLM identifies relevant tables before searching
- ✅ **Count-aware retrieval** — adjusts vector search depth for count/total queries
- ✅ **Anti-hallucination** — answers only from retrieved context
- ✅ **Qdrant Cloud** — persistent vector index, auto-builds on first run
- ✅ **REST API** — plug into any frontend or workflow

---

## 📁 Project Structure

```
rag-sql-chatbot/
├── RAG_REST.py        # Main FastAPI app
├── requirements.txt   # Dependencies
├── .env.example       # Environment variable template
└── README.md
```

---

## 🔧 Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/rag-sql-chatbot.git
cd rag-sql-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env`:
```env
QDRANT_API_KEY=your_qdrant_api_key
GOOGLE_API_KEY=your_google_gemini_api_key
```

### 4. Run the server
```bash
python RAG_REST.py
```

API will be live at: `http://localhost:8000`

---

## 📡 API Usage

### POST `/chatbot/chat`

```bash
curl -X POST http://localhost:8000/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many active job postings are there?"}'
```

**Response:**
```json
{
  "answer": "There are currently 47 active job postings in the database."
}
```

---

## 🌍 Use Cases

- HR teams querying recruitment databases
- Non-technical users accessing business data
- Internal analytics without BI tools
- Any domain with a structured SQL database

---

## 📌 Requirements

```
fastapi
uvicorn
langchain
langchain-community
langchain-google-genai
qdrant-client
sentence-transformers
sqlalchemy
pyodbc
pandas
```

---

## 👤 Author

**Thayanban Thamizhendhal**  
Python Developer | AI/ML Engineer  
[LinkedIn](https://linkedin.com/in/thayanbanthamizhendhal) · AWS Certified AI Practitioner
