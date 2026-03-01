import streamlit as st
import os
from graph import graph_app  # This is the new import for LangGraph

# 1. Page Setup
st.set_page_config(page_title="PDF AI Assistant", page_icon="📄")
st.title("📄 Chat with your PDF")

# 2. Initialize chat history so it remembers the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display chat history from the session
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. User Input Area
if prompt := st.chat_input("Ask me something about your PDF..."):
    # Show the user's message in the chat
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- THE LANGGRAPH ENGINE ---
    with st.spinner("LangGraph is thinking..."):
        try:
            # Send the question to your Graph workflow
            # This calls 'retrieve' then 'assistant' nodes automatically
            result = graph_app.invoke({"question": prompt})
            
            # Extract the final answer from the graph's State
            full_response = result["answer"]
            
            # Show the AI's response
            st.chat_message("assistant").markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Something went wrong: {e}")