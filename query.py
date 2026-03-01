import chromadb

def query_rag(query_text):
    # Connect to the database you already built
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="langchain")
    
    # Search for the 3 most relevant parts of the PDF
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )
    
    # Join the findings into one block of text
    context_text = "\n\n---\n\n".join(results['documents'][0])
    return context_text