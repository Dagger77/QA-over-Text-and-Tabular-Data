"""
Document ingestion pipeline for LightRAG.
Parses RTF files and stores embeddings using OpenAI models.
"""

import os
import asyncio
from pathlib import Path

import dotenv
from striprtf.striprtf import rtf_to_text
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# ----------------------------
# Configuration
# ----------------------------
dotenv.load_dotenv()
DOCUMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledgebase-docs"))


# ----------------------------
# Document Utilities
# ----------------------------
def load_rtf_documents(directory: str) -> list[str]:
    """
    Loads all .rtf files from the given directory and extracts plain text.
    """
    texts = []
    folder = Path(directory)

    for file_path in folder.glob("*.rtf"):
        if not file_path.is_file():
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                rtf_content = f.read()
                text = rtf_to_text(rtf_content)
                texts.append(text)
        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")

    return texts


# ----------------------------
# RAG Ingestion Logic
# ----------------------------
async def ingest_documents():
    """
    Initialize LightRAG and ingest all extracted RTF document texts.
    """
    print("Initializing LightRAG...")

    rag = LightRAG(
        working_dir=DOCUMENTS_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    docs = load_rtf_documents(DOCUMENTS_DIR)
    print(f"Loaded {len(docs)} RTF documents.")

    for i, doc in enumerate(docs, start=1):
        await rag.ainsert(doc)
        print(f"Document {i} ingested.")

    print("All documents successfully ingested into LightRAG.")


if __name__ == "__main__":
    asyncio.run(ingest_documents())
