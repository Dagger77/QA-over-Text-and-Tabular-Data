from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from pydantic_ai import Agent

from rag_agent import run_rag_agent
from sql_agent import run_sql_agent
from summary_agent import run_summary_agent


# Define shared state
class AgentState(TypedDict):
    input: str
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


async def route(state: AgentState) -> Literal["rag", "sql", "hybrid"]:
    intent = await intent_router.run(state["input"])
    return intent.data.strip().lower()


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
builder.add_node("sql", sql_node)
builder.add_node("rag", rag_node)
builder.add_node("summarize", summarize_node)

# Routing logic
builder.add_conditional_edges("input", route, {
    "sql": "sql",
    "rag": "rag",
    "hybrid": "rag" # for hybrid, we'll trigger rag first, then go to sql manually
})

# Link to summarization
builder.add_edge("sql", "summarize")
builder.add_edge("rag", "summarize")
builder.add_edge("hybrid", "sql")  # after rag runs in hybrid, go to sql

# End
builder.add_edge("summarize", END)

# Compile graph
multi_agent_graph = builder.compile()
