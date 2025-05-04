import pandas as pd
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding


# Load CSV
df = pd.read_csv("Job-Description.csv")


# Convert rows into documents
documents = []
for _, row in df.iterrows():
    row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(Document(text=row_text))

# Use Ollama for LLM and Embeddings (no OpenAI involved)
llm = Ollama(model="llama3", base_url="http://localhost:11434", request_timeout=300)  # You can also try "llama2" or "llama3"
embed_model = OllamaEmbedding(model_name="llama3")  # You can also try "nomic-embed-text" or "mxbai-embed-large"


# Build the index with both models
index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)

# Create a query engine
query_engine = index.as_query_engine(llm=llm, embed_model=embed_model)

# Ask a question
response = query_engine.query("Can you write me the names of the persons in the csv file?")
print(response)