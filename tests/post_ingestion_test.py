import sqlite3
import pytest
import os

from tests.utils import missing_required_tables


@pytest.fixture(scope="module")
def db_connection():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "student_data.db"))
    conn = sqlite3.connect(db_path)
    yield conn
    conn.close()


def test_tables_exist(db_connection):
    cursor = db_connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = {row[0] for row in cursor.fetchall()}
    assert "student_info_basic" in table_names
    assert "student_info_detailed" in table_names


def test_sample_row_count(db_connection):
    cursor = db_connection.execute("SELECT COUNT(*) FROM student_info_basic")
    count = cursor.fetchone()[0]
    assert count > 0


@pytest.mark.asyncio
async def test_sql_agent_math_score_query():
    from agents.sql_agent import run_sql_agent
    result = await run_sql_agent("average math score of students who completed test preparation")
    assert "avg" in str(result).lower() or "mathscore" in str(result).lower()


@pytest.mark.asyncio
async def test_rag_agent_returns_known_term():
    from agents.rag_agent import run_rag_agent, initialize_rag
    rag = await initialize_rag()
    result = await run_rag_agent("What is STEM?", lightrag=rag)
    assert "science" in result.lower() and "technology" in result.lower()


@pytest.mark.asyncio
async def test_hybrid_query_summary():
    from orchestration.orchestration import multi_agent_graph
    state = {"input": "How does parental education affect math scores?"}
    final_state = await multi_agent_graph.ainvoke(state)
    assert "final_answer" in final_state
    assert "math" in final_state["final_answer"].lower()


@pytest.mark.asyncio
async def test_long_sql_output_truncation():
    from agents.sql_agent import run_sql_agent
    result = await run_sql_agent("list all student records")
    assert len(result["rows"]) <= 5


@pytest.mark.asyncio
async def test_valid_sql_query():
    from agents.sql_agent import run_sql_agent
    result = await run_sql_agent("How many female students scored above 90 in math?")
    assert "rows" in result
    assert isinstance(result["rows"], list)


@pytest.mark.asyncio
async def test_sql_query_with_invalid_column():
    from agents.sql_agent import run_sql_agent
    result = await run_sql_agent("How many students have a pet?")
    assert "error" in result or "explanation" in result
