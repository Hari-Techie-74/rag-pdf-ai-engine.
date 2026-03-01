import chromadb

def query_rag(query_text):
    # This matches your folder name in the screenshot
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="langchain")
    
    results = collection.query(
        query_texts=[query_text],
        n_results=1
    )
    
    if results['documents']:
        return results['documents'][0][0]
    return "No answer found."