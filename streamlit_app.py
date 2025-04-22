from dotenv import load_dotenv
import streamlit as st
import asyncio
from datetime import datetime
import time

from orchestration import multi_agent_graph
from summary_agent import summary_agent

load_dotenv()

DEBUG_MODE = True
LOG_TO_FILE = True
LOG_FILE_PATH = "agent_logs.txt"


async def main():
    st.title("Multi-Agent Knowledge Explorer")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the documents or data...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Retrieving the answer...")

            start = time.time()
            state = {"input": user_input}
            final_state = await multi_agent_graph.ainvoke(state)
            end = time.time()

            outputs = []
            if final_state.get("rag_output"):
                outputs.append(final_state["rag_output"])
            if final_state.get("sql_output"):
                outputs.append(final_state["sql_output"])
            combined_input = "\n\n".join(f"Answer {i + 1}: {out}" for i, out in enumerate(outputs))

            full_response = ""
            async with summary_agent.run_stream(combined_input) as stream:
                async for delta in stream.stream_text(delta=True):
                    full_response += delta
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

            if LOG_TO_FILE:
                with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                    f.write("\n=== Log Entry @ {} ===\n".format(datetime.now().isoformat()))
                    f.write(f"Question: {user_input}\n")
                    f.write(f"Final Answer: {full_response}\n")
                    f.write(f"Response Time: {end - start:.2f} seconds\n")
                    if "rag_output" in final_state:
                        f.write(f"RAG Output: {final_state['rag_output']}\n")
                    if "sql_output" in final_state:
                        f.write(f"SQL Output: {final_state['sql_output']}\n")
                    f.write("============================\n")

            if DEBUG_MODE:
                with st.expander("🔍 Debug Info"):
                    st.markdown("**Intent:** " + final_state.get("intent", "unknown"))
                    st.markdown("**Response Time:** {:.2f} seconds".format(end - start))
                    st.markdown("**RAG Output**")
                    st.markdown(final_state.get("rag_output", "_None_"))

                    st.markdown("**SQL Output**")
                    st.markdown(final_state.get("sql_output", "_None_"))

                    if final_state.get("sql_output") and "**Query:**" in final_state["sql_output"]:
                        lines = final_state["sql_output"].splitlines()
                        query_line = next((line for line in lines if line.startswith("**Query:**")), "")
                        explanation = final_state["sql_output"].split("**Answer:**")[0].replace(query_line, "")
                        st.markdown("**SQL Query**")
                        st.code(query_line.replace("**Query:** ", ""))
                        st.markdown("**Explanation**")
                        st.markdown(explanation.strip())


if __name__ == "__main__":
    asyncio.run(main())
