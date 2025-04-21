import os
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import dotenv
from striprtf.striprtf import rtf_to_text
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv()

documents_path = "./knowledgebase-docs"


def load_documents(doc_path: str):
    """
    Loads all .rtf files in directory and returns list of plain text content.

    Returns:
        str: Extracted plain text.
    """
    text_list = []
    folder = Path(doc_path)

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue  # Skip directories and non-files

        ext = file_path.suffix.lower()

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
                text = rtf_to_text(rtf_content)
                text_list.append(text)

        except Exception as e:
            print(f'Failed to load {file_path.name}: {e}')
            continue

    return text_list


async def initialize_rag():
    rag = LightRAG(
        working_dir=documents_path,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance and ingest docs
    rag = asyncio.run(initialize_rag())
    print('rag initialized')
    docs = load_documents(documents_path)
    print('Docs are ready')
    for doc in docs:
        rag.insert(doc)


if __name__ == "__main__":
    main()