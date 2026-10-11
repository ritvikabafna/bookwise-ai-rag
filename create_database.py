
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the PDF
file_path = (
    Path(__file__).parent
    / "Documents loader"
    / "The Love Hypothesis - Ali Hazelwood.pdf"
)

docs = PyPDFLoader(str(file_path)).load()

print("PDF pages:", len(docs))

# 2. Split the PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))

# 3. Load the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Set the database directory
db_path = Path(__file__).parent / "chroma-db-v2"

# 5. Remove the old database collection, if present
if db_path.exists():
    try:
        old_db = Chroma(
            collection_name="love_hypothesis",
            embedding_function=embedding_model,
            persist_directory=str(db_path),
        )
        old_db.delete_collection()
        print("Old collection deleted.")
    except Exception as e:
        print("Could not delete old collection:", e)

# 6. Create a fresh database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(db_path),
    collection_name="love_hypothesis",
)

print("\nDatabase rebuilt successfully!")
print("Stored chunks:", vectorstore._collection.count())
