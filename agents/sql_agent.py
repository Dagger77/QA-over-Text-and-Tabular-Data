"""
Pydantic AI agent that builds SQL queries and retrieves data from a local SQLite database.
"""

import os
import sqlite3
import asyncio
from dataclasses import dataclass
from typing import Annotated, Union, TypedDict, List, Dict

from annotated_types import MinLen
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry, format_as_xml

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "student_data.db")

# Schema for guiding the agent
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
    rows: List[Dict] = Field(default_factory=list)


class InvalidRequest(BaseModel):
    error_message: str


class SQLAgentOutput(TypedDict, total=False):
    sql_query: str
    explanation: str
    rows: List[Dict]
    error: str


Response = Union[Success, InvalidRequest]

agent = Agent[SQLDeps, Response](
    model="openai:gpt-4o-mini",
    deps_type=SQLDeps,
    output_type=Response,
    instrument=True,
)


def get_table_columns(conn: sqlite3.Connection, table: str) -> set:
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return {row[1].lower() for row in cursor.fetchall()}
    except Exception:
        return set()


def get_categorical_value_hints(conn: sqlite3.Connection) -> str:
    """
    Generate a string summary of distinct values for key categorical columns.
    """
    categorical_columns = {
        "student_info_basic": ["Gender", "EthnicGroup", "LunchType", "TestPrep", "ParentEduc"],
        "student_info_detailed": ["ParentMaritalStatus", "PracticeSport", "IsFirstChild", "TransportMeans",
                                  "WklyStudyHours"]
    }

    hints = []
    for table, columns in categorical_columns.items():
        for col in columns:
            try:
                cursor = conn.execute(f"SELECT DISTINCT {col} FROM {table}")
                values = sorted(set(str(row[0]) for row in cursor.fetchall() if row[0] is not None))
                formatted = ", ".join(values)
                hints.append(f"- {table}.{col}: {formatted}")
            except Exception:
                continue
    return "\n".join(hints)


@agent.system_prompt
async def system_prompt(ctx: RunContext[SQLDeps]) -> str:
    value_hints = get_categorical_value_hints(ctx.deps.conn)
    return f"""
You are an assistant that helps translate user questions into SQL queries for a SQLite database.

The database contains two related tables that may have overlapping or complementary data:
- student_info_detailed
- student_info_basic

If a question can be answered using data from both tables, generate two separate SELECT queries — one for each table — and run them both. Present both results clearly and separately.
Do not use UNION. Assume the tables contain distinct, possibly inconsistent data.

Refer to these distinct categorical values while forming the query:
{value_hints}

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

    conn = ctx.deps.conn
    used_query = output.sql_query.lower()

    basic_cols = get_table_columns(conn, "student_info_basic")
    detailed_cols = get_table_columns(conn, "student_info_detailed")
    all_columns = basic_cols.union(detailed_cols)

    used_columns = {col for col in all_columns if col in used_query}

    # Fix invalid column references
    if "student_info_basic" in used_query and not used_columns.issubset(basic_cols):
        output.explanation += "\n\nNote: some columns aren't in 'student_info_basic'. Switched to 'student_info_detailed'."
        output.sql_query = output.sql_query.replace("student_info_basic", "student_info_detailed")

    # Duplicate query for both tables if needed
    if "student_info_basic" in output.sql_query:
        detailed_query = output.sql_query.replace("student_info_basic", "student_info_detailed")
        output.sql_query += f";\n{detailed_query}"

    rows = []
    errors = []
    for query in output.sql_query.split(";"):
        query = query.strip()
        if not query:
            continue
        try:
            cursor = conn.execute(query)
            fetched = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
            rows.extend([dict(zip(cols, row)) for row in fetched])
        except sqlite3.Error as e:
            errors.append(f"Error for query `{query}`: {e}")

    output.rows = rows[:5]  # limit output for token safety
    if errors:
        output.explanation += "\n\nSome queries failed:\n" + "\n".join(errors)

    return output


async def run_sql_agent(question: str) -> SQLAgentOutput:
    conn = sqlite3.connect(DB_PATH)
    deps = SQLDeps(conn)

    result = await agent.run(question, deps=deps)
    conn.close()

    if isinstance(result.output, Success):
        return {
            "sql_query": result.output.sql_query,
            "explanation": result.output.explanation,
            "rows": result.output.rows,
        }
    elif isinstance(result.output, InvalidRequest):
        return {"error": result.output.error_message}
    else:
        return {"error": "Unknown error occurred."}


async def main():
    user_input = input("Ask a question: ")
    result = await run_sql_agent(user_input)

    print("\n--- SQL Agent Result ---")
    if "rows" in result:
        print("Query:", result["sql_query"])
        print("Explanation:", result.get("explanation", ""))
        print("Answer:")
        for row in result["rows"]:
            print(row)
    elif "error" in result:
        print("Error:", result["error"])


if __name__ == "__main__":
    asyncio.run(main())
