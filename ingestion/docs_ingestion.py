import os
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import dotenv
from striprtf.striprtf import rtf_to_text
from pathlib import Path

dotenv.load_dotenv()

documents_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledgebase-docs"))


def load_documents(doc_path: str):
    """
    Loads all .rtf files in directory and returns list of plain text content.
    """
    text_list = []
    folder = Path(doc_path)

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue  # Skip directories and non-files

        ext = file_path.suffix.lower()
        if ext != ".rtf":
            continue

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
                text = rtf_to_text(rtf_content)
                text_list.append(text)

        except Exception as e:
            print(f'Failed to load {file_path.name}: {e}')
            continue

    return text_list


async def ingest_documents():
    """Initialize LightRAG and ingest all RTF documents."""
    rag = LightRAG(
        working_dir=documents_path,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    docs = load_documents(documents_path)
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        await rag.ainsert(doc)
    print("Documents ingested into RAG successfully.")


if __name__ == "__main__":
    asyncio.run(ingest_documents())
