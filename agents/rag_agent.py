"""
Pydantic AI agent that leverages LightRAG for question answering over local documents.
"""

import os
import sys
import time
import argparse
import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

load_dotenv()

DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledgebase-docs")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)


# LightRAG Setup
async def initialize_rag() -> LightRAG:
    """
    Create and initialize a LightRAG instance with OpenAI models.
    """
    rag = LightRAG(
        working_dir=DOCUMENTS_PATH,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    print("LightRAG initialized.")
    return rag


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    lightrag: LightRAG


# Agent Definition
agent = Agent(
    model="openai:gpt-4o-mini",
    deps_type=RAGDeps,
    system_prompt="You are a helpful assistant that answers questions about how demographic factors influence student performance based on the provided documents. "
                  "Use the retrieve tool to get relevant information from documents before answering. "
                  "Reply in short, crispy manner. "
                  "Prefer the document content over general knowledge. "
                  "If the question is not related to the documents topic, mention it in the reply. "
                  "In case when the documents doesn't contain the answer, clearly state that the information isn't available"
                  "in the current documents and provide your best general knowledge response."
)


@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str) -> str:
    """
    Retrieve relevant context from the documents using LightRAG.
    """
    return await context.deps.lightrag.aquery(
        search_query, param=QueryParam(mode="mix")
    )


async def run_rag_agent(question: str, lightrag: LightRAG) -> str:
    """
    Run the RAG agent with the provided question and LightRAG instance.
    """
    start = time.time()
    deps = RAGDeps(lightrag=lightrag)
    result = await agent.run(question, deps=deps)
    duration = time.time() - start
    print(f"RAG execution time: {duration:.2f}s")
    return result.output


async def main():
    parser = argparse.ArgumentParser(description="Query the LightRAG knowledgebase.")
    parser.add_argument("--question", help="The question to answer")
    args = parser.parse_args()

    if not args.question:
        print("Please provide a question using --question.")
        return

    rag = await initialize_rag()
    answer = await run_rag_agent(args.question, lightrag=rag)

    print("\nResponse:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
