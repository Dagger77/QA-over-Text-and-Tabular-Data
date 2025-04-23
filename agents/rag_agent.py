"""Pydantic AI agent that leverages RAG with a local LightRAG for question answering."""

import os
import sys
import time
import argparse
from dataclasses import dataclass
import asyncio

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

dotenv.load_dotenv()

DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledgebase-docs")

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)


async def initialize_rag():
    """Create and initialize LightRAG."""
    rag = LightRAG(
        working_dir=DOCUMENTS_PATH,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )

    await rag.initialize_storages()

    print("LightRAG initialized")

    return rag


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    lightrag: LightRAG


# Create the Pydantic AI agent
agent = Agent(
    model='openai:gpt-4o-mini',
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
    """Retrieve relevant documents using LightRAG."""
    return await context.deps.lightrag.aquery(
        search_query, param=QueryParam(mode="naive", top_k=3)
    )


async def run_rag_agent(question: str, lightrag: LightRAG) -> str:
    """Run the RAG agent with a provided LightRAG instance."""
    start = time.time()
    deps = RAGDeps(lightrag=lightrag)
    result = await agent.run(question, deps=deps)
    print(f"RAG total time: {time.time() - start:.2f}s")
    return result.output


async def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with LightRAG")
    parser.add_argument("--question", help="The question to answer about knowledgebase documents")
    args = parser.parse_args()

    rag = await initialize_rag()
    answer = await run_rag_agent(args.question, lightrag=rag)

    print("\nResponse:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
