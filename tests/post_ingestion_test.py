"""
Post-ingestion integration tests for RAG, SQL, and multi-agent orchestration.
"""

import os
import sqlite3
import pytest


# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture(scope="module")
def db_connection():
    """Provides a SQLite DB connection for table validation tests."""
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "student_data.db"))
    conn = sqlite3.connect(db_path)
    yield conn
    conn.close()


# ---------------------------
# SQLite Ingestion Tests
# ---------------------------

def test_tables_exist(db_connection):
    """Verify expected tables exist after ingestion."""
    cursor = db_connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = {row[0] for row in cursor.fetchall()}
    assert "student_info_basic" in table_names
    assert "student_info_detailed" in table_names


def test_basic_table_has_data(db_connection):
    """Ensure the basic table is not empty."""
    cursor = db_connection.execute("SELECT COUNT(*) FROM student_info_basic")
    count = cursor.fetchone()[0]
    assert count > 0


# ---------------------------
# SQL Agent Tests
# ---------------------------

@pytest.mark.asyncio
async def test_valid_sql_query():
    from agents.sql_agent import run_sql_agent
    result = await run_sql_agent("How many female students scored above 90 in math?")
    assert "rows" in result
    assert isinstance(result["rows"], list)


@pytest.mark.asyncio
async def test_sql_average_math_score():
    from agents.sql_agent import run_sql_agent
    result = await run_sql_agent("average math score of students who completed test preparation")
    result_str = str(result).lower()
    assert "avg" in result_str or "mathscore" in result_str


@pytest.mark.asyncio
async def test_sql_invalid_column_handled():
    from agents.sql_agent import run_sql_agent
    result = await run_sql_agent("How many students have a pet?")
    assert "error" in result or "explanation" in result


@pytest.mark.asyncio
async def test_sql_row_limit():
    from agents.sql_agent import run_sql_agent
    result = await run_sql_agent("list all student records")
    assert len(result["rows"]) <= 5


# ---------------------------
# RAG Agent Test
# ---------------------------

@pytest.mark.asyncio
async def test_rag_agent_known_term():
    from agents.rag_agent import run_rag_agent, initialize_rag
    rag = await initialize_rag()
    result = await run_rag_agent("What is STEM?", lightrag=rag)
    result_lower = result.lower()
    assert "science" in result_lower and "technology" in result_lower


# ---------------------------
# Hybrid Flow Test
# ---------------------------

@pytest.mark.asyncio
async def test_hybrid_summary_generation():
    from orchestration.orchestration import multi_agent_graph
    state = {"input": "How does parental education affect math scores?"}
    final_state = await multi_agent_graph.ainvoke(state)

    assert "final_answer" in final_state
    assert "math" in final_state["final_answer"].lower()
