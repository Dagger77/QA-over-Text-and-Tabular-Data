"""Pydantic AI agent that builds SQL query and retrieves data from local database."""
import os
import sqlite3
import asyncio
from dataclasses import dataclass
from typing import Annotated, Union, TypedDict
from datetime import date

from annotated_types import MinLen
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry, format_as_xml

import dotenv

dotenv.load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "student_data.db")

# Schema definition for prompt (SQLite style)
DB_SCHEMA = """
CREATE TABLE student_info_detailed (
    Gender TEXT,
    EthnicGroup TEXT,
    ParentEduc TEXT,
    LunchType TEXT,
    TestPrep TEXT,
    ParentMaritalStatus TEXT,
    PracticeSport TEXT,
    IsFirstChild TEXT,
    NrSiblings INTEGER,
    TransportMeans TEXT,
    WklyStudyHours TEXT,
    MathScore INTEGER,
    ReadingScore INTEGER,
    WritingScore INTEGER
);

CREATE TABLE student_info_basic (
    Gender TEXT,
    EthnicGroup TEXT,
    ParentEduc TEXT,
    LunchType TEXT,
    TestPrep TEXT,
    MathScore INTEGER,
    ReadingScore INTEGER,
    WritingScore INTEGER
);
"""

# Example queries for guiding the agent
SQL_EXAMPLES = [
    {
        'request': 'average math score of students who completed test preparation',
        'response': "SELECT AVG(MathScore) FROM student_info_detailed WHERE TestPrep = 'completed'"
    },
    {
        'request': 'how many students are first children',
        'response': "SELECT COUNT(*) FROM student_info_detailed WHERE IsFirstChild = 'yes'"
    },
    {
        'request': 'list of students who scored above 90 in reading',
        'response': "SELECT * FROM student_info_basic WHERE ReadingScore > 90"
    },
    {
        'request': 'How many female students got more than 90 in math?',
        'response': (
            "SELECT COUNT(*) AS count_basic FROM student_info_basic WHERE Gender = 'female' AND MathScore > 90;\n"
            "SELECT COUNT(*) AS count_detailed FROM student_info_detailed WHERE Gender = 'female' AND MathScore > 90;"
        )
    }
]


@dataclass
class SQLDeps:
    conn: sqlite3.Connection


class Success(BaseModel):
    sql_query: Annotated[str, MinLen(1)]
    explanation: str = Field('', description='Explanation of the SQL query')
    rows: list[dict] = Field(default_factory=list)


class InvalidRequest(BaseModel):
    error_message: str


class SQLAgentOutput(TypedDict, total=False):
    sql_query: str
    explanation: str
    rows: list[dict]
    error: str


Response = Union[Success, InvalidRequest]

agent: Agent[SQLDeps, Response] = Agent(
    model='openai:gpt-4o-mini',
    deps_type=SQLDeps,
    output_type=Response,
    instrument=True,
)

def get_table_columns(conn, table_name):
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return {row[1].lower() for row in cursor.fetchall()}
    except Exception:
        return set()

@agent.system_prompt
async def system_prompt() -> str:
    return f"""
You are an assistant that helps translate user questions into SQL queries for a SQLite database.

The database contains two related tables that may have overlapping or complementary data:
- student_info_detailed
- student_info_basic

If a question can be answered using data from both tables, generate two separate SELECT queries — one for each table — and run them both. Present both results clearly and separately.
Do not use UNION. Assume the tables contain distinct, possibly inconsistent data.

Here is the schema:
{DB_SCHEMA}

Here are some examples:
{format_as_xml(SQL_EXAMPLES)}
"""


@agent.output_validator
async def validate_output(ctx: RunContext[SQLDeps], output: Response) -> Response:
    if isinstance(output, InvalidRequest):
        return output

    output.sql_query = output.sql_query.replace('\\', '')
    if not output.sql_query.strip().upper().startswith("SELECT"):
        raise ModelRetry("Please generate SELECT queries only.")

    # Schema-aware correction
    detailed_columns = get_table_columns(ctx.deps.conn, "student_info_detailed")
    basic_columns = get_table_columns(ctx.deps.conn, "student_info_basic")
    used_query = output.sql_query.lower()
    used_columns = {col for col in detailed_columns.union(basic_columns) if col in used_query}

    if "student_info_basic" in used_query:
        if not used_columns.issubset(basic_columns):
            output.explanation += "\n\nNote: The query referenced columns not present in 'student_info_basic'. Switched to 'student_info_detailed'."
            output.sql_query = output.sql_query.replace("student_info_basic", "student_info_detailed")

    # Attempt to handle inconsistency by duplicating if both apply
    if "student_info_basic" in output.sql_query:
        detailed_query = output.sql_query.replace("student_info_basic", "student_info_detailed")
        output.sql_query += f";\n{detailed_query}"

    rows = []
    errors = []
    for query in output.sql_query.strip().split(';'):
        query = query.strip()
        if not query:
            continue
        try:
            cursor = ctx.deps.conn.execute(query)
            rows.extend([
                dict(zip([col[0] for col in cursor.description], row))
                for row in cursor.fetchall()
            ])
        except sqlite3.Error as e:
            errors.append(f"Error for query `{query}`: {str(e)}")

    # handles max token limit, limits number of rows in output
    MAX_ROWS = 5
    rows = rows[: MAX_ROWS]

    output.rows = rows
    if errors:
        output.explanation += "\n\nSome queries failed:\n" + "\n".join(errors)

    return output


async def run_sql_agent(question: str) -> SQLAgentOutput:
    conn = sqlite3.connect(os.path.abspath(DB_PATH))
    deps = SQLDeps(conn)

    result = await agent.run(question, deps=deps)
    conn.close()

    if hasattr(result.output, "rows") and result.output.rows:
        return {
            "sql_query": result.output.sql_query,
            "explanation": result.output.explanation,
            "rows": result.output.rows,
        }
    elif hasattr(result.output, "explanation"):
        return {"explanation": result.output.explanation}
    elif hasattr(result.output, "error_message"):
        return {"error": result.output.error_message}
    else:
        return {"error": "No answer returned."}


async def main():
    user_query = input("Ask a question about the data: ")
    result = await run_sql_agent(user_query)

    print("\n--- Result ---")
    if "rows" in result:
        print("\n**Query:**", result["sql_query"])
        print("\n**Answer:**")
        for row in result["rows"]:
            print(row)
    elif "explanation" in result:
        print(result["explanation"])
    elif "error" in result:
        print("Error:", result["error"])


if __name__ == "__main__":
    asyncio.run(main())
