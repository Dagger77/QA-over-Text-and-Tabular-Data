"""A summarizer agent that combines multiple sources (e.g., SQL + RAG) into a human-friendly response"""

import os
import sys
from pydantic_ai import Agent

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)

summary_agent = Agent(
    model="openai:gpt-4o",
    system_prompt=(
        "You are a summarizer agent. Your job is to take multiple answers from RAG and SQL agents, and create a clear, concise, and natural response for the user. "
        "Focus on clarity, avoid raw SQL or formatting, and make it sound like one coherent assistant response. "
        "If you receive only one input, simply rephrase it nicely."
    )
)


async def run_summary_agent(agent_outputs: list[str]) -> str:
    """
    Combine and rephrase the responses from SQL and/or RAG agents.
    """
    combined_input = "\n\n".join(f"Answer {i + 1}: {out}" for i, out in enumerate(agent_outputs))
    result = await summary_agent.run(combined_input)
    return result.output
