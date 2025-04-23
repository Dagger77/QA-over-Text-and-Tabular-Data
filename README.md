# 📊 Multi-Agent QA over Text and Tabular Data

This project is a multi-agent Streamlit application that can answer questions using:
- 🗃️ Structured tabular data (via SQL generation)
- 📄 Unstructured documents (via Retrieval-Augmented Generation, RAG) 
- 🧠 Combined (hybrid) reasoning with summarization

---

## 🚀 Features
- **LangGraph orchestration** for routing queries to SQL, RAG, or both
- **Pydantic AI agents** for structured SQL output and RAG responses
- **LightRAG** with local vector database for fast document retrieval (https://lightrag.github.io/)
- **Streaming UI** with chat history, debug info, and logs
- **Automatic data ingestion** for both CSV and RTF sources

---

## 🧰 Project Structure
```
.
├── app/
│   └── streamlit_app.py          # Main app UI
├── agents/
│   ├── sql_agent.py              # SQL generator and executor
│   ├── rag_agent.py              # RAG interface with LightRAG
│   └── summary_agent.py          # Summarizer agent for hybrid output
├── orchestration/
│   └── orchestration.py          # LangGraph workflow with routing
├── ingestion/
│   ├── table_ingestion.py        # Loads CSV into SQLite
│   └── docs_ingestion.py         # Ingests RTF files into LightRAG
├── data/
│   ├── knowledgebase-data/       # CSV files
│   └── knowledgebase-docs/       # RTF documents
├── student_data.db               # SQLite database (auto-generated)
├── .env                          # OpenAI API key
├── requirements.txt              # Python dependencies
└── tests/                        # Test suite
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Dagger77/QA-over-Text-and-Tabular-Data.git
cd QA-over-Text-and-Tabular-Data
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add OpenAI key
Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE = 'https://api.openai.com/v1' # for some reason required for LightRag
```

---

## 🧪 Running Tests
```bash
pytest
```
> Includes ingestion checks, agent verification, and hybrid orchestration tests.

---

## 🧠 Run the App
```bash
streamlit run app/streamlit_app.py
```

---

## 💡 Example Questions
- What is inclusive education?
- How does lunch type affect math scores?
- Average reading score of students who completed test prep
- How many students are first children?

---

## 🧩 Future Enhancements
- Visualisation of tabular data
