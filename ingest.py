import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DOCS_DIR = "docs"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def main():
    print(f"Loading documents from {DOCS_DIR} folder...")
    
    # 1. Load Documents
    # We map specific file extensions to their respective Langchain loaders
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader
    }
    
    documents = []
    
    # Iterate through docs folder directly and use appropriate loader
    if not os.path.exists(DOCS_DIR):
        print(f"Error: Directory '{DOCS_DIR}' not found. Please create it and add some documents.")
        return

    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        ext = os.path.splitext(filename)[1].lower()
        if ext in loaders:
            print(f"Loading {filename}...")
            loader_class = loaders[ext]
            loader = loader_class(file_path)
            try:
                docs = loader.load()
                documents.extend(docs)
                print(f"  - Loaded {len(docs)} pages/parts from {filename}")
            except Exception as e:
                print(f"  - Error loading {filename}: {e}")
        else:
            print(f"Skipping {filename}: unsupported file extension ({ext})")
            
    if not documents:
        print("No documents loaded. Please check if your files are supported (.pdf or .docx).")
        return

    print(f"\nTotal documents loaded: {len(documents)}")

    # 2. Split Text 
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Embeddings & Vector Store
    print(f"Initializing HuggingFace embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"Storing chunks in Chroma DB at {CHROMA_DB_DIR}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print("\nIngestion Complete! The vector database is ready.")

if __name__ == "__main__":
    main()
