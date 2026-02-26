import chromadb
from chromadb.config import Settings

# 1. Connect to the database folder you already created
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Get the collection (usually named "documents" or "langchain")
collection = client.get_collection(name="langchain")

# 3. Ask your question
question = input("What is your question about the documents? ")

# 4. Search for the answer
results = collection.query(
    query_texts=[question],
    n_results=3
)

# 5. Show the results
print("\n--- Found in your documents ---")
for doc in results['documents'][0]:
    print(f"- {doc[:300]}...")