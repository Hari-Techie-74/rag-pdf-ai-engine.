import streamlit as st
import os
from query import query_rag  # This connects the UI to your RAG logic

st.set_page_config(page_title="PDF AI Assistant", page_icon="📄")
st.title("📄 Chat with your PDF")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me something about your PDF..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- BRAIN SURGERY HAPPENS HERE ---
    with st.spinner("Searching PDF and thinking..."):
        try:
            # We call your actual RAG function from query.py
            response = query_rag(prompt)
        except Exception as e:
            response = f"Error: {str(e)}"
    
    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})