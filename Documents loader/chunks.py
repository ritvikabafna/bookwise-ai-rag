
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

# Load the PDF
file_path = Path(__file__).parent / "SwarSatya_Login.pdf"

data = PyPDFLoader(str(file_path))

docs = data.load()

# Split the PDF text into chunks
splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

# Print the total number of chunks
print("Total chunks:", len(chunks))

# Print the first chunk
if chunks:
    print("\nFirst chunk:")
    print(chunks[0].page_content)
