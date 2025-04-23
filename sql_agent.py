"""Pydantic AI agent that builds SQL query and retrieves data from local database."""
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

    output.rows = rows
    if errors:
        output.explanation += "\n\nSome queries failed:\n" + "\n".join(errors)

    return output

async def run_sql_agent(question: str) -> SQLAgentOutput:
    conn = sqlite3.connect("student_data.db")
    deps = SQLDeps(conn)

    modified_question = question.strip() + (
        "\n\nIMPORTANT: You MUST return the same SELECT query for both 'student_info_basic' and 'student_info_detailed'. Run each SELECT separately. Do NOT return only one query."
    )

    result = await agent.run(modified_question, deps=deps)
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
