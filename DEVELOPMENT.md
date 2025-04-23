# 🛠️ Development Log — QA-over-Text-and-Tabular-Data

Chronological changelog of major features, improvements, and decisions.

---

## v1.0

- **LightRAG implemented** for document-based question answering
- **Text2SQL agent implemented** to query SQLite from natural language
- **LangGraph orchestration** connecting RAG + SQL + summarization
- **Streamlit App UI** integrated with agent framework

---

## v1.01

- **top_k parameter updated** in LightRAG to improve performance
- **Naive mode enabled** in LightRAG for faster document search
- **Intent classifier prompt** improved to better route complex queries

---

## v1.02

- **Tabular Data inconsistency detected** — SQL agent now queries both tables
- **Summarizer upgraded** to `gpt-4o` for faster output 
- **Auto-ingestion added**: database and RAG index initialized on first app launch

---

## v1.03

- Minor **prompt tuning and UI layout polish**
- **Test suite expanded**, covering RAG, SQL, and hybrid queries
- **Large SQL result edge case handled** via row truncation
- Project **README.md created and updated**

---

📘 See `README.md` for usage instructions and `tests/` for test coverage.
