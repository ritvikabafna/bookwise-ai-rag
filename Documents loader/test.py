
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# 1. Load the text file
file_path = Path(__file__).parent / "note.txt"

loader = TextLoader(str(file_path), encoding="utf-8")
docs = loader.load()

text = docs[0].page_content

print("Text file loaded successfully!")
print("Original text length:", len(text))

# 2. Create the prompt
template = PromptTemplate.from_template(
    """You are an expert summarizer.
Summarize the following text in 5-7 bullet points.
Include only the main ideas. Do not copy the entire text.

TEXT:
{text}

SUMMARY:"""
)

# 3. Load the local Qwen model
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 250,
        "do_sample": False,
        "return_full_text": False,
    },
)

# 4. Combine the prompt and text
prompt = template.format(text=text)

# 5. Generate the summary
print("\nGenerating summary...\n")

result = llm.invoke(prompt)

print("SUMMARY:")
print(result)
