"""Pydantic AI agent that leverages RAG with a local LightRAG for question answering."""

import os
import sys
import argparse
from dataclasses import dataclass
import asyncio

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from openai import AsyncOpenAI

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# Load environment variables from .env file
dotenv.load_dotenv()

documents_path = "./knowledgebase-docs"

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)


async def initialize_rag():
    rag = LightRAG(
        working_dir=documents_path,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )

    await rag.initialize_storages()

    return rag


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    lightrag: LightRAG


# Create the Pydantic AI agent
agent = Agent(
    model='openai:gpt-4o-mini',
    deps_type=RAGDeps,
    system_prompt="You are a helpful assistant that answers questions about how demographic factors influence student performance based on the provided documents."
                  "Use the retrieve tool to get relevant information from documents before answering."
                  "Reply in short, crispy manner."
                  "Prefer the document content over general knowledge."
                  "If the question is not related to the documents topic, mention it in the reply"
                  "In case when the documents doesn't contain the answer, clearly state that the information isn't available"
                  "in the current documents and provide your best general knowledge response."
)


@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str) -> str:
    """Retrieve relevant documents.

    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.

    Returns:
        Formatted context information from the retrieved documents.
    """
    return await context.deps.lightrag.aquery(
        search_query, param=QueryParam(mode="mix")
    )


async def run_rag_agent(question: str, ) -> str:
    """Run the RAG agent to answer a question about knowledgebase documents.

    Args:
        question: The question to answer.

    Returns:
        The agent's response.
    """
    # Create dependencies
    lightrag = await initialize_rag()
    deps = RAGDeps(lightrag=lightrag)

    # Run the agent
    result = await agent.run(question, deps=deps)

    return result.data


def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using ChromaDB")
    parser.add_argument("--question", help="The question to answer about knowledgebase documents")

    args = parser.parse_args()

    # Run the agent
    response = asyncio.run(run_rag_agent(args.question))

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
