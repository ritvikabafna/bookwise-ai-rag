
---
title: BookWise AI
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
short_description: Chat with PDF books using Retrieval-Augmented Generation
---

# 📚 BookWise AI — Chat with Your Books Using RAG

BookWise AI is an AI-powered application that lets users upload PDF books and ask questions about their contents.

## ✨ Features

- 📄 Upload PDF books through a Streamlit interface
- 🔍 Retrieve relevant document chunks using semantic search
- 🧠 Generate answers using Qwen 2.5 1.5B Instruct
- 🗂️ Store document embeddings using ChromaDB
- 📖 Display retrieved source snippets and page references
- 📚 Maintain a separate vector database for each uploaded PDF

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- ChromaDB
- Qwen 2.5 1.5B Instruct
- PyPDF

## ⚙️ How It Works

1. Upload a PDF book.
2. Extract text from its pages.
3. Split the text into overlapping chunks.
4. Generate embeddings using MiniLM.
5. Retrieve relevant chunks from ChromaDB.
6. Generate an answer using Qwen and the retrieved context.

## 🚀 Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## ⚠️ Limitations

- Supports PDF files in the current version.
- Scanned PDFs may require OCR.
- Answer quality depends on text extraction and retrieval relevance.
- Running Qwen locally or on CPU-based hosting can be slow.

## 🔮 Future Improvements

- Support for EPUB and DOCX
- Conversation memory
- Improved retrieval and reranking
- Chapter summaries and quiz generation

## 👩‍💻 Project

**BookWise AI** — Chat with your books using Retrieval-Augmented Generation (RAG).
