import streamlit as st
import pandas as pd
import os
import sys
import logging
import re
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JobRecommender")

# Check for required libraries
try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.llms.ollama import Ollama
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
except ImportError:
    st.error("Required packages not installed. Please install with: pip install llama-index pandas streamlit")
    st.stop()


class JobRecommender:
    def __init__(self, csv_path="Job-Description.csv", model_name="llama3.1", debug_mode=False):
        """
        Initialize the Job Recommender with a CSV file and Llama model.
        
        Args:
            csv_path: Path to CSV file containing job descriptions
            model_name: Name of the Ollama model to use
            debug_mode: Enable detailed debugging output
        """
        self.csv_path = csv_path
        self.model_name = model_name
        self.debug_mode = debug_mode
        self.df = None
        self.index = None
        self.raw_documents = []
        self.ollama_available = False
        
        # Load data immediately
        self._load_data()
        
        # Try to set up LlamaIndex with Ollama
        try:
            self.llm = Ollama(model=model_name, request_timeout=120.0)
            self.embedding_model = OllamaEmbedding(model_name=model_name)
            
            # Configure LlamaIndex settings
            Settings.llm = self.llm
            Settings.embed_model = self.embedding_model
            self.ollama_available = True
            logger.info("LLM and embedding model configured successfully")
            
            # Build index only if Ollama is available
            self._build_index()
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            logger.warning("Continuing with keyword search only")
        
    def _load_data(self):
        """Load job descriptions from CSV file with extensive error checking."""
        logger.info(f"Loading data from {self.csv_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(self.csv_path):
                logger.error(f"File not found: {self.csv_path}")
                return False
                
            # Try reading the CSV
            self.df = pd.read_csv(self.csv_path)
            
            # Debug information
            logger.info(f"CSV loaded successfully with {len(self.df)} rows")
            logger.info(f"Columns found: {list(self.df.columns)}")
            
            # Clean up column names and values
            self.df = self.df.rename(columns={col: col.strip() for col in self.df.columns})
            
            # Clean up string values and handle missing data
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].fillna('').astype(str).str.strip()
            
            # Check for required columns
            required_columns = ['Name', 'Job Title', 'Job Description']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            return True
                
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return False
    
    def _normalize_text(self, text):
        """Clean and normalize text by removing extra spaces."""
        # Replace multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()
        
    def _build_index(self):
        """Build a vector index from the job descriptions with fallback for direct keyword matching."""
        logger.info("Building vector index and preparing search data...")
        
        # Create documents from the job descriptions
        documents = []
        
        for idx, row in self.df.iterrows():
            # Use all available fields from your CSV
            name = row.get('Name', '')
            role = row.get('Job Title', '')
            description = row.get('Job Description', '')
            division = row.get('Division', '') if 'Division' in self.df.columns else ''
            tags = row.get('Tags', '') if 'Tags' in self.df.columns else ''
            
            # Clean up text fields
            description = self._normalize_text(description)
            tags = self._normalize_text(tags)
            
            # Create a comprehensive text document
            text = f"Name: {name}\nJob Title: {role}\nDivision: {division}\n"
            text += f"Job Description: {description}\nTags/Skills: {tags}\n"
            
            # Store the raw document for direct keyword matching
            self.raw_documents.append({
                "id": idx,
                "name": name,
                "role": role,
                "division": division,
                "description": description,
                "tags": tags,
                "text": text.lower()  # Lowercase for easier matching
            })
            
            # Create document for vector indexing
            doc = Document(text=text, metadata={
                "id": idx, 
                "name": name,
                "tags": tags,
                "role": role,
                "division": division
            })
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} documents for indexing")
        
        try:
            # Split text into smaller chunks for better retrieval
            text_splitter = SentenceSplitter(chunk_size=512)
            nodes = text_splitter.get_nodes_from_documents(documents)
            logger.info(f"Split into {len(nodes)} nodes")
            
            # Create vector store index
            self.index = VectorStoreIndex(nodes)
            logger.info("Vector index built successfully!")
            return True
        except Exception as e:
            logger.error(f"Error building index: {e}")
            logger.warning("Continuing with fallback keyword search only")
            return False
    
    def _keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform direct keyword matching as fallback.
        
        Args:
            query: String containing search terms
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents based on keyword matching
        """
        logger.info(f"Performing keyword search for: {query}")
        
        query_terms = query.lower().split()
        results = []
        
        for doc in self.raw_documents:
            # Calculate simple relevance score based on term frequency
            score = 0
            for term in query_terms:
                # Check tags with higher weight
                if term in doc["tags"].lower():
                    score += 10  # Higher weight for tag matches
                
                # Check description
                if term in doc["description"].lower():
                    score += 1
                    
                # Check role
                if term in doc["role"].lower():
                    score += 1
            
            if score > 0:
                results.append({
                    "name": doc["name"],
                    "details": doc["text"],
                    "relevance_score": score / len(query_terms),  # Normalize by query length
                    "tags": doc["tags"]
                })
        
        # Sort by score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        logger.info(f"Keyword search found {len(results)} matches")
        return results[:top_k]
        
    def get_recommended_contacts(self, interests: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get recommended job contacts based on user interests, with fallbacks.
        
        Args:
            interests: String describing user interests and skills
            top_k: Number of top results to return
            
        Returns:
            List of recommended contacts with relevance scores
        """
        logger.info(f"Finding matches for interests: {interests}")
        vector_results = []
        
        # Try vector search first if index is available
        if self.index and self.ollama_available:
            try:
                # Create retriever with higher top_k
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=min(top_k * 3, len(self.df))  # Triple requested but cap at max entries
                )
                
                # Set up query engine with much lower similarity cutoff to ensure results
                query_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)]  # Very low threshold
                )
                
                # Generate query from interests
                query = f"""
                Based on these interests and skills, find relevant job contacts:
                {interests}
                
                Identify people whose job descriptions or tags match these interests.
                """
                
                # Execute query
                response = query_engine.query(query)
                
                # Process vector search results
                for node in response.source_nodes:
                    name = node.metadata.get('name', 'Unknown')
                    text = node.text
                    score = node.score if hasattr(node, 'score') else 0
                    tags = node.metadata.get('tags', '')
                    
                    # Boost score for direct tag matches
                    interest_terms = interests.lower().split()
                    tag_terms = tags.lower().split()
                    
                    tag_match_bonus = 0
                    for term in interest_terms:
                        if any(term in tag for tag in tag_terms):
                            tag_match_bonus += 0.2
                    
                    vector_results.append({
                        "name": name,
                        "details": text,
                        "relevance_score": score + tag_match_bonus,
                        "tags": tags,
                        "source": "vector"
                    })
                
                logger.info(f"Vector search found {len(vector_results)} results")
                
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                logger.info("Falling back to keyword search only")
        
        # Always perform keyword search as fallback or supplement
        keyword_results = self._keyword_search(interests, top_k)
        
        # If we got vector results, combine them with keyword results
        if vector_results:
            # Mark keyword results
            for result in keyword_results:
                result["source"] = "keyword"
            
            # Combine and deduplicate by name
            combined_results = vector_results + keyword_results
            unique_names = set()
            final_results = []
            
            for result in combined_results:
                if result["name"] not in unique_names:
                    unique_names.add(result["name"])
                    final_results.append(result)
            
            # Sort by score
            final_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return final_results[:top_k]
        else:
            # Just use keyword results
            return keyword_results


# Streamlit UI setup
def main():
    st.set_page_config(
        page_title="Matchmakers @HQ",
        page_icon="👔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Matchmakers @HQ")
    st.subheader("Find the right people based on your needs")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Fixed CSV path - "Job-Description.csv"
        csv_path = "Job-Description.csv"
        
        # Ollama model selection
        model_options = ["llama3.1","mistral"]
        selected_model = st.selectbox("Select Ollama Model", model_options)
        
        # Number of results
        top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=3)
        
        # Debug mode toggle
        #debug_mode = st.checkbox("Enable Debug Mode", value=False)
        
        st.markdown("---")
        st.markdown("""
        ### How to use
        1. Choose the model
        2. Enter your needs or interests in the main panel
        3. View recommended contacts
        """)
    
    # Main panel
    # Initialize recommender immediately with fixed CSV path
    try:
        recommender = JobRecommender(csv_path, selected_model, debug_mode=False)
        
        if recommender.df is not None:
            #st.success(f"Using job data with {len(recommender.df)} entries")
            st.success(f"Using job data with 47 entries")
        else:
            st.error(f"Error loading CSV from {csv_path}. Please check if the file exists.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error initializing job recommender: {str(e)}")
        st.stop()
    
    # System status
    st.markdown("---")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.subheader("System Status")
        st.write(f"CSV File: ✅ Using {csv_path} ({len(recommender.df)} entries)")
        vector_status = "✅ Available" if recommender.index else "❌ Not available"
        st.write(f"Vector Index: {vector_status}")
    
    with status_col2:
        ollama_status = "✅ Connected" if recommender.ollama_available else "❌ Not connected"
        st.write(f"Ollama Service: {ollama_status}")
        st.write(f"Model: {selected_model}")
        search_method = "Vector + Keyword" if recommender.index and recommender.ollama_available else "Keyword only"
        st.write(f"Search Method: {search_method}")
    
    st.markdown("---")
    
    # Interest input
    st.subheader("Who do you want to meet?")
    with st.form("interest_form"):
        interests = st.text_area("Enter your needs or interests", 
                                placeholder="e.g. Innovation, Data Science, Policy Planning")
        submit_button = st.form_submit_button("Find Recommended Contacts")
    
    # Search on form submission
    if submit_button and interests:
        with st.spinner("Searching for matches..."):
            results = recommender.get_recommended_contacts(interests, top_k)
            
        if results:
            st.subheader(f"Top {len(results)} Recommended Contacts")
            
            for i, rec in enumerate(results, 1):
                with st.expander(f"Match #{i}: {rec['name']} (Score: {rec['relevance_score']:.2f})"):
                    # Format the details for better readability
                    details_lines = rec['details'].split('\n')
                    for line in details_lines:
                        if line.strip():  # Skip empty lines
                            key, *value_parts = line.split(':', 1)
                            value = value_parts[0] if value_parts else ""
                            st.markdown(f"**{key.strip()}:** {value.strip()}")
                    
                    # Show match method if available
                    if 'source' in rec:
                        search_icon = "🔍" if rec['source'] == "vector" else "🔑"
                        st.caption(f"{search_icon} Match method: {rec['source']}")
        else:
            st.warning("No suitable matches found. Try different search terms.")
            
            # # Debug info on failure
            # if debug_mode:
            #     st.subheader("Debug Information")
            #     st.write(f"CSV has {len(recommender.df)} entries")
            #     st.write(f"Available columns: {list(recommender.df.columns)}")
                
            #     # Show all available tags for debugging
            #     if 'Tags' in recommender.df.columns:
            #         all_tags = set()
            #         for tags in recommender.df['Tags']:
            #             tag_list = [t.strip() for t in tags.split()]
            #             all_tags.update(tag_list)
            #         st.write(f"All available tags: {', '.join(sorted(all_tags))}")

if __name__ == "__main__":
    main()
