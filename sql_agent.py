"""Pydantic AI agent that builds SQL query and retrieves data from local database."""
import sqlite3
import asyncio
from dataclasses import dataclass
from typing import Annotated, Union
from datetime import date

from annotated_types import MinLen
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry, format_as_xml

import dotenv

dotenv.load_dotenv()

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


Response = Union[Success, InvalidRequest]

agent: Agent[SQLDeps, Response] = Agent(
    model='openai:gpt-4o-mini',
    deps_type=SQLDeps,
    output_type=Response,
    instrument=True,
)


@agent.system_prompt
async def system_prompt() -> str:
    return f"""
You are an assistant that helps translate user questions into SQL queries for a SQLite database.

Today's date is {date.today()}.

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
        raise ModelRetry("Please generate a SELECT query.")

    try:
        cursor = ctx.deps.conn.execute(output.sql_query)
        output.rows = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        raise ModelRetry(f"Invalid SQL query: {e}") from e

    return output


async def run_sql_agent(question: str) -> str:
    """
    Run the SQL agent to answer a question using SQLite data.

    Args:
        question: Natural language question

    Returns:
        String answer with result rows or explanation
    """
    # Connect to the SQLite DB
    conn = sqlite3.connect("student_data.db")
    deps = SQLDeps(conn)

    # Run the agent
    result = await agent.run(question, deps=deps)

    # Close connection
    conn.close()

    # Format result
    if hasattr(result.output, "rows") and result.output.rows:
        rows = result.output.rows
        return f"**Query:** `{result.output.sql_query}`\n\n**Answer:**\n" + "\n".join(str(r) for r in rows)
    elif hasattr(result.output, "explanation"):
        return result.output.explanation
    elif hasattr(result.output, "error_message"):
        return f"Error: {result.output.error_message}"
    else:
        return "No answer returned."


def main():
    conn = sqlite3.connect("student_data.db")
    deps = SQLDeps(conn)

    user_query = input("Ask a question about the data: ")

    result = asyncio.run(run_sql_agent(user_query))

    print("\n--- SQL Query ---")
    print(result.output.sql_query)
    print("\n--- Explanation ---")
    print(result.output.explanation)
    print("\n--- Result Rows ---")
    for row in result.output.rows:
        print(row)

    conn.close()


if __name__ == "__main__":
    main()
