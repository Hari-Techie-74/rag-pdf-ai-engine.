from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama # The Local Engine
from query import query_rag

class State(TypedDict):
    question: str
    context: str
    answer: str

def retrieval_node(state: State):
    content = query_rag(state["question"])
    return {"context": content}

def assistant_node(state: State):
    # This now talks to the Llama 3 running on YOUR Mac
    llm = ChatOllama(model="llama3")
    
    prompt = f"""
    You are a helpful assistant. Use the following PDF context to answer the question.
    Context: {state['context']}
    Question: {state['question']}
    """
    response = llm.invoke(prompt)
    return {"answer": response.content}

# Build the Graph
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("assistant", assistant_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "assistant")
workflow.add_edge("assistant", END)

graph_app = workflow.compile()