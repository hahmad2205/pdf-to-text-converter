import streamlit as st
from rag import app

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["first_call"] = True

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    config = {"configurable": {"thread_id": "thread-11"}}

    with st.spinner("Thinking..."):
        response = app(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response.content})
    st.chat_message("assistant").write(response.content)
else:
    st.chat_message("assistant").write("Please enter a prompt to proceed.")
