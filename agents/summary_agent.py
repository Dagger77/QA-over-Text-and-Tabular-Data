"""
A summarizer agent that combines multiple responses (e.g., SQL and RAG) into a natural, user-friendly answer.
"""

import os
import sys
from typing import List

from pydantic_ai import Agent

if not os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY is not set in the environment.")
    sys.exit(1)


# Summarizer Agent Configuration
summary_agent = Agent(
    model="openai:gpt-4o",
    system_prompt=(
        "You are a summarizer agent. Your job is to take multiple answers from RAG and SQL agents, and create a clear, concise, and natural response for the user. "
        "Focus on clarity, avoid raw SQL or formatting, and make it sound like one coherent assistant response. "
        "If you receive only one input, simply rephrase it nicely."
    )
)


async def run_summary_agent(agent_outputs: List[str]) -> str:
    """
    Combine and rephrase the responses from SQL and/or RAG agents into one answer.
    """
    combined = "\n\n".join(f"Answer {i+1}: {text}" for i, text in enumerate(agent_outputs))
    result = await summary_agent.run(combined)
    return result.output
