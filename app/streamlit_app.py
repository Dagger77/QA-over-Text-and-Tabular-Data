import os
import sys
import time
import sqlite3
import asyncio
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Ensure import paths for agents/orchestration work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.rag_agent import initialize_rag
from orchestration.orchestration import multi_agent_graph
from agents.summary_agent import summary_agent
from ingestion.table_ingestion import load_data
from ingestion.docs_ingestion import ingest_documents

# ---------------------------
# Config & Paths
# ---------------------------
load_dotenv()
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "agent_logs.txt")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "student_data.db")
DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledgebase-docs")


# ---------------------------
# Init Check Functions
# ---------------------------
def is_sqlite_initialized() -> bool:
    """Check if both tables exist in SQLite DB."""
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        return {"student_info_basic", "student_info_detailed"}.issubset(tables)
    except Exception:
        return False


def is_rag_initialized() -> bool:
    """Check if required vector files exist in RAG storage."""
    required_files = ["vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json"]
    return all(os.path.exists(os.path.join(DOCS_PATH, file)) for file in required_files)


# ---------------------------
# Streamlit App Logic
# ---------------------------
async def main():
    st.set_page_config(page_title="Multi-Agent QA", layout="centered")
    st.title("Multi-Agent Knowledge Explorer")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        debug_mode = st.checkbox("Enable Debug Mode", value=False)
        log_to_file = st.checkbox("Log Responses to File", value=False)
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # One-time setup
    if not is_sqlite_initialized():
        with st.spinner("Initializing SQLite database..."):
            load_data()
        st.toast("Table data is ready!")

    if not is_rag_initialized():
        with st.spinner("Initializing RAG knowledgebase..."):
            await ingest_documents()
        st.toast("Documents are ready!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag" not in st.session_state:
        st.session_state.rag = await initialize_rag()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about the documents or data...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Generating the answer..."):
                start = time.time()

                # Run graph
                state = {"input": user_input, "rag_instance": st.session_state.rag}
                final_state = await multi_agent_graph.ainvoke(state)
                end = time.time()

                # Combine outputs for summarization
                outputs = []
                if final_state.get("rag_output"):
                    outputs.append(final_state["rag_output"])
                if final_state.get("sql_output"):
                    outputs.append(final_state["sql_output"])

                # Stream summarization
                full_response = ""
                combined_input = "\n\n".join(f"Answer {i+1}: {o}" for i, o in enumerate(outputs))
                async with summary_agent.run_stream(combined_input) as stream:
                    async for delta in stream.stream_text(delta=True):
                        full_response += delta
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Log
                if log_to_file:
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        f.write(f"\n=== Log Entry @ {datetime.now().isoformat()} ===\n")
                        f.write(f"Question: {user_input}\n")
                        f.write(f"Final Answer: {full_response}\n")
                        f.write(f"Response Time: {end - start:.2f} seconds\n")
                        if "rag_output" in final_state:
                            f.write(f"RAG Output: {final_state['rag_output']}\n")
                        if "sql_output" in final_state:
                            f.write(f"SQL Output: {final_state['sql_output']}\n")
                        f.write("============================\n")

                # Debug UI
                if debug_mode:
                    with st.expander("Debug Info"):
                        st.markdown(f"**Intent:** `{final_state.get('intent', 'unknown')}`")
                        st.markdown(f"**Response Time:** `{end - start:.2f} seconds`")
                        if final_state.get("rag_output"):
                            st.markdown("**RAG Output**")
                            st.code(final_state["rag_output"], language="markdown")
                        if final_state.get("sql_output"):
                            st.markdown("**SQL Output**")
                            st.code(final_state["sql_output"], language="markdown")


if __name__ == "__main__":
    asyncio.run(main())
