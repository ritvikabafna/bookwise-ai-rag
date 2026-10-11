
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)

# 1. Load the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Open the existing Chroma database
db_path = Path(__file__).parent / "chroma-db-v2"

vectorstore = Chroma(
    collection_name="love_hypothesis",
    embedding_function=embedding_model,
    persist_directory=str(db_path),
)

print("Stored chunks:", vectorstore._collection.count())

# 3. Configure the retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8},
)

# 4. Load Qwen
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 150,
        "do_sample": False,
        "return_full_text": False,
    },
)

print("\nRAG system created!")
print("Ask questions about the book. Type 0 to exit.")

# 5. Ask questions continuously
while True:
    query = input("\nYou: ").strip()

    if query == "0":
        break

    if not query:
        continue

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    print("\n--- Retrieved documents ---")

    for i, doc in enumerate(docs, start=1):
        print(f"\nChunk {i}:")
        print(doc.page_content[:500])
        print("Metadata:", doc.metadata)

    print("\n--- End of retrieved documents ---")

    # 6. Build a plain-text prompt for Qwen
    final_prompt = f"""
You answer questions about The Love Hypothesis.

Use only the information in the context below.
Answer the question directly in your own words.
If the context does not contain the answer, say:
"I could not find the answer in the document."

Context:
{context}

Question: {query}

Answer:
"""

    # 7. Generate the answer
    response = llm.invoke(final_prompt)

    print("\nAI:", response.strip())
