#!/bin/bash

echo "Setting up RAG Job Recommender environment..."

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install pandas llama-index ollama
pip install llama_index.embeddings.ollama
pip install llama_index.llms.ollama

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed successfully!"
else
    echo "Ollama is already installed."
fi

# Pull the Llama 3.1 model
echo "Pulling Llama 3.1 model... (this may take a while)"
ollama pull llama3.1

echo "Setup complete! You can now run the job recommender."
echo "Usage: python job_recommender.py [path_to_csv]"
