"""
Orchestration logic for Multi-Agent system using LangGraph.
Coordinates RAG, SQL, and summarization agents via intent classification.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from lightrag import LightRAG
from pydantic_ai import Agent

from agents.rag_agent import run_rag_agent
from agents.sql_agent import run_sql_agent
from agents.summary_agent import run_summary_agent


# ----------------------------
# Shared State Definition
# ----------------------------
class AgentState(TypedDict):
    input: str
    intent: str
    rag_output: str
    sql_output: str
    final_answer: str
    rag_instance: LightRAG


# ----------------------------
# Intent Classifier Agent
# ----------------------------
intent_router = Agent(
    model="openai:gpt-4o-mini",
    system_prompt=(
        "You are an intent classifier for a multi-agent system.\n\n"
        "Given a user's question, respond with just one word:\n"
        "- `sql` → if the question asks for patterns, averages, trends, group comparisons, or any analysis of structured data in tables.\n"
        "- `rag` → if the question asks for definitions, context, or information that can be found in documents.\n"
        "- `hybrid` → if both documents and data are needed to answer the question.\n\n"
        "Examples:\n"
        "- \"What is the average reading score by lunch type?\" → sql\n"
        "- \"How does lunch type affect performance?\" → sql\n"
        "- \"What are the common parental education levels?\" → sql\n"
        "- \"Why is parental education important?\" → rag\n"
        "- \"Show me data and explanation about lunch impact\" → hybrid\n\n"
        "Available tables and columns:\n"
        "- student_info_basic(Gender, EthnicGroup, ParentEduc, LunchType, TestPrep, MathScore, ReadingScore, WritingScore)\n"
        "- student_info_detailed(Gender, EthnicGroup, ParentEduc, LunchType, TestPrep, ParentMaritalStatus, PracticeSport, IsFirstChild, NrSiblings, TransportMeans, WklyStudyHours, MathScore, ReadingScore, WritingScore)\n\n"
        "Consider this schema when deciding if a question involves structured data."
    )
)


async def classify_node(state: AgentState) -> AgentState:
    """Classifies intent using LLM."""
    response = await intent_router.run(state["input"])
    state["intent"] = response.output.strip().lower()
    return state


def decide_next_step(state: AgentState) -> str:
    """Determines which node to route to based on intent."""
    return state["intent"]


# ----------------------------
# SQL Node
# ----------------------------
async def sql_node(state: AgentState) -> AgentState:
    result = await run_sql_agent(state["input"])

    if "error" in result:
        state["sql_output"] = f"Error: {result['error']}"
    else:
        parts = []
        if result.get("sql_query"):
            parts.append(f"**Query:** `{result['sql_query']}`")
        if result.get("explanation"):
            parts.append(f"**Explanation:**\n{result['explanation']}")
        if result.get("rows"):
            parts.append("**Answer:**\n" + "\n".join(str(r) for r in result["rows"]))
        else:
            parts.append("**Answer:**\n_No rows returned._")
        state["sql_output"] = "\n\n".join(parts)

    return state


# ----------------------------
# RAG Node
# ----------------------------
async def rag_node(state: AgentState) -> AgentState:
    state["rag_output"] = await run_rag_agent(state["input"], lightrag=state["rag_instance"])
    return state


# ----------------------------
# Summarization Node
# ----------------------------
async def summarize_node(state: AgentState) -> AgentState:
    outputs = []
    if state.get("rag_output"):
        outputs.append(state["rag_output"])
    if state.get("sql_output"):
        outputs.append(state["sql_output"])

    state["final_answer"] = await run_summary_agent(outputs)
    return state


# ----------------------------
# LangGraph Definition
# ----------------------------
builder = StateGraph(AgentState)

builder.set_entry_point("classify")
builder.add_node("classify", classify_node)
builder.add_node("sql", sql_node)
builder.add_node("rag", rag_node)
builder.add_node("summarize", summarize_node)

# Conditional routing from classifier
builder.add_conditional_edges("classify", decide_next_step, {
    "sql": "sql",
    "rag": "rag",
    "hybrid": "rag"  # hybrid starts with RAG, then branches to SQL
})

# Conditional routing after RAG node
builder.add_conditional_edges("rag", lambda s: "sql" if s["intent"] == "hybrid" else "summarize", {
    "sql": "sql",
    "summarize": "summarize"
})

# Always summarize at the end
builder.add_edge("sql", "summarize")
builder.add_edge("summarize", END)

# Compile the graph
multi_agent_graph = builder.compile()