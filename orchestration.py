from langgraph.graph import StateGraph, END
from typing import TypedDict
from pydantic_ai import Agent

from rag_agent import run_rag_agent
from sql_agent import run_sql_agent
from summary_agent import run_summary_agent


# Define shared state
class AgentState(TypedDict):
    input: str
    intent: str
    rag_output: str
    sql_output: str
    final_answer: str


# LLM-based intent classifier
intent_router = Agent(
    model="openai:gpt-4o-mini",
    system_prompt=(
        "You are a classifier. Given a user's question, respond with one word:\n"
        "- 'sql' if it requires querying structured data\n"
        "- 'rag' if it asks for information from documents\n"
        "- 'hybrid' if it needs both"
    )
)


async def classify_node(state: AgentState) -> AgentState:
    intent = await intent_router.run(state["input"])
    state["intent"] = intent.data.strip().lower()
    return state


def decide_next_step(state: AgentState) -> str:
    return state["intent"]


# SQL agent node
async def sql_node(state: AgentState) -> AgentState:
    state["sql_output"] = await run_sql_agent(state["input"])
    return state


# RAG agent node
async def rag_node(state: AgentState) -> AgentState:
    state["rag_output"] = await run_rag_agent(state["input"])
    return state


# Summary node
async def summarize_node(state: AgentState) -> AgentState:
    outputs = []
    if state.get("rag_output"):
        outputs.append(state["rag_output"])
    if state.get("sql_output"):
        outputs.append(state["sql_output"])

    state["final_answer"] = await run_summary_agent(outputs)
    return state


# Define the LangGraph
builder = StateGraph(AgentState)

builder.add_node("classify", classify_node)
builder.add_node("sql", sql_node)
builder.add_node("rag", rag_node)
builder.add_node("summarize", summarize_node)

builder.set_entry_point("classify")

# Routing after classification
builder.add_conditional_edges("classify", decide_next_step, {
    "sql": "sql",
    "rag": "rag",
    "hybrid": "rag"  # hybrid starts with RAG, then branches to SQL
})

# After RAG, route depending on intent
builder.add_conditional_edges("rag", lambda s: "sql" if s["intent"] == "hybrid" else "summarize", {
    "sql": "sql",
    "summarize": "summarize"
})

builder.add_edge("sql", "summarize")
builder.add_edge("summarize", END)

# Compile graph
multi_agent_graph = builder.compile()
