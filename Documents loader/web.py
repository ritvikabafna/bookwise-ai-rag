import os
os.environ["USER_AGENT"] = "RAGLearningProject/1.0"

from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

url = "https://www.apple.com/in/macbook-pro/"

data= WebBaseLoader(url)

docs= data.load()
print(docs[0].page_content)