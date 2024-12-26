#configurable_rag.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai
from pathlib import Path
import faiss
import pickle
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import re
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SearchMode(Enum):
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

@dataclass
class RAGConfig:
    """Configuration class for RAG system."""
    search_mode: SearchMode
    initial_k: int  # Number of documents to retrieve before reranking
    final_k: int  # Number of documents after reranking
    use_reranking: bool = True
    bm25_weight: float = 0.3  # Only used in hybrid mode
    semantic_weight: float = 0.7  # Only used in hybrid mode
    
    def __post_init__(self):
        if self.initial_k < self.final_k:
            raise ValueError("initial_k must be greater than or equal to final_k")
        if self.search_mode == SearchMode.HYBRID:
            if not (0 <= self.bm25_weight <= 1) or not (0 <= self.semantic_weight <= 1):
                raise ValueError("Weights must be between 0 and 1")
            if abs(self.bm25_weight + self.semantic_weight - 1) > 1e-6:
                raise ValueError("Weights must sum to 1")

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]

def preprocess_text(text: str) -> str:
    """Preprocess text for better matching."""
    # Handle camelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Handle special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Convert to lowercase and normalize whitespace
    text = ' '.join(text.lower().split())
    return text

class ConfigurableRAG:
    def __init__(
        self, 
        schema_file: str,
        openai_api_key: str, 
        config: RAGConfig,
        index_dir: str = "index"
    ):
        """Initialize configurable RAG system."""
        self.config = config
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        print("Loading models...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        if config.use_reranking:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.embedding_dim = 384
        
        # Initialize paths
        self.index_path = self.index_dir / "faiss_index.idx"
        self.docs_path = self.index_dir / "documents.pkl"
        self.bm25_path = self.index_dir / "bm25.pkl"
        
        # Load or create indices
        if self._indices_exist():
            print("Loading existing indices and documents...")
            self._load_indices_and_documents()
        else:
            print(f"Loading unified schema from {schema_file}...")
            with open(schema_file, 'r') as f:
                self.schema = json.load(f)
            
            print("Creating documents...")
            schema_docs = self._create_documents()
            
            # Add system information
            from anura_system_info import SystemInformation
            system_info = SystemInformation()
            self.documents = schema_docs + system_info.get_docs()
            print(f"Created {len(self.documents)} documents")
            
            print("Building indices...")
            self._build_and_save_indices()

    def _indices_exist(self) -> bool:
        """Check if required index files exist based on configuration."""
        required_files = [self.index_path, self.docs_path]
        if self.config.search_mode == SearchMode.HYBRID:
            required_files.append(self.bm25_path)
        return all(f.exists() for f in required_files)

    def _load_indices_and_documents(self):
        """Load indices based on configuration."""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        if self.config.search_mode == SearchMode.HYBRID:
            with open(self.bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
                self.bm25 = bm25_data['index']
                self.tokenized_corpus = bm25_data['tokenized_corpus']

    def _build_and_save_indices(self):
        """Build and save indices based on configuration."""
        # Prepare corpus with preprocessed text
        corpus = [preprocess_text(doc.content) for doc in self.documents]
        raw_corpus = [doc.content for doc in self.documents]
        
        # Build FAISS index
        print("Building FAISS index...")
        embeddings = self.embedding_model.encode(raw_corpus, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(normalized_embeddings.astype(np.float32))
        
        # Build BM25 index if needed
        if self.config.search_mode == SearchMode.HYBRID:
            print("Building BM25 index...")
            self.tokenized_corpus = [word_tokenize(doc) for doc in corpus]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            bm25_data = {
                'index': self.bm25,
                'tokenized_corpus': self.tokenized_corpus
            }
            with open(self.bm25_path, 'wb') as f:
                pickle.dump(bm25_data, f)
        
        # Save FAISS index and documents
        print("Saving indices...")
        faiss.write_index(self.index, str(self.index_path))
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)

    def _create_documents(self) -> List[Document]:
        """Create document chunks from unified schema with comprehensive field information."""
        documents = []
        
        for table_name, table_info in self.schema.items():
            # Create field documents with enhanced content
            for field in table_info["columns"]:
                # Add field-specific content
                field_content = [
                    f"Table: {table_name}",
                    f"Field: {field['Column Name']}",
                    f"In table {table_name}, the field {field['Column Name']} has the following properties:"
                ]
                
                # Add all field properties with better formatting
                for key, value in field.items():
                    if key != "Column Name" and value is not None and str(value) != "NaN":
                        if key == "Explanation":
                            field_content.append(f"Purpose and Usage: {value}")
                        else:
                            field_content.append(f"{key}: {value}")
                
                # Add searchable combinations
                field_content.append(f"{table_name}.{field['Column Name']}")  # Add dot notation
                field_content.append(f"{field['Column Name']} in {table_name}")  # Add natural language
                
                content = "\n".join(field_content)
                
                field_doc = Document(
                    content=content,
                    metadata={
                        "table": table_name,
                        "field": field["Column Name"],
                        "type": "field",
                        "table_type": table_info["type"],  # Add the table type (Input/Output)
                        **{k: v for k, v in field.items() if v is not None and str(v) != "NaN"}
                    }
                )
                documents.append(field_doc)
            
            # Create enhanced table overview document
            table_fields = [field["Column Name"] for field in table_info["columns"]]

            # Get required fields
            required_fields = [field["Column Name"] for field in table_info["columns"] 
                             if field.get("Required") == "Yes"]

            # Get primary key fields
            primary_key_fields = [field["Column Name"] for field in table_info["columns"] 
                                if field.get("Primary Key") == "Yes"]  # Note: using "Primary Key" instead of "PK"
            
            table_content = [
                f"Table: {table_name}",
                f"Type: {table_info['type']}",  # Add table type (Input/Output)
                f"Description: {table_info['description']}",
                f"This table contains the following fields: {', '.join(table_fields)}",
                f"Primary key fields are: {', '.join(primary_key_fields) if primary_key_fields else 'None'}",
                f"Required fields are: {', '.join(required_fields) if required_fields else 'None'}",
                f"Total number of fields: {len(table_info['columns'])}"
            ]

            table_doc = Document(
                content="\n".join(table_content),
                metadata={
                    "table": table_name,
                    "type": "table",
                    "table_type": table_info["type"],  # Add the table type (Input/Output)
                    "description": table_info["description"],
                    "fields": table_fields,
                    "primary_key_fields": primary_key_fields,
                    "required_fields": required_fields
                }
            )
            documents.append(table_doc)
        
        return documents

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0,1] range."""
        if len(scores) == 0:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score != min_score:
            return (scores - min_score) / (max_score - min_score)
        return np.ones_like(scores)

    def _semantic_search(self, query: str) -> Tuple[List[int], np.ndarray]:
        """Perform semantic search using FAISS."""
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        similarities, indices = self.index.search(query_embedding, self.config.initial_k)
        normalized_similarities = self._normalize_scores(similarities[0])
        
        return indices[0], normalized_similarities

    def _lexical_search(self, query: str) -> Tuple[List[int], np.ndarray]:
        """Perform lexical search using BM25."""
        # Preprocess query
        processed_query = preprocess_text(query)
        tokenized_query = word_tokenize(processed_query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[-self.config.initial_k:][::-1]
        top_k_scores = scores[top_k_indices]
        
        # Normalize scores
        normalized_scores = self._normalize_scores(top_k_scores)
        
        return top_k_indices, normalized_scores

    def _hybrid_search(self, query: str) -> List[Tuple[Document, float]]:
        """Combine semantic and lexical search results with normalized scores."""
        semantic_indices, semantic_scores = self._semantic_search(query)
        lexical_indices, lexical_scores = self._lexical_search(query)
        
        seen_indices = set()
        hybrid_results = []
        
        all_indices = list(semantic_indices) + list(lexical_indices)
        for idx in all_indices:
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            
            # Get semantic and lexical scores (0 if document wasn't in top-k)
            sem_idx = np.where(semantic_indices == idx)[0]
            lex_idx = np.where(lexical_indices == idx)[0]
            
            semantic_score = semantic_scores[sem_idx[0]] if len(sem_idx) > 0 else 0
            lexical_score = lexical_scores[lex_idx[0]] if len(lex_idx) > 0 else 0
            
            # Combine scores using configured weights
            hybrid_score = (self.config.semantic_weight * semantic_score + 
                          self.config.bm25_weight * lexical_score)
            
            hybrid_results.append((self.documents[idx], hybrid_score))
        
        # Sort by score and return top-k
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        return hybrid_results[:self.config.initial_k]

    def _rerank(self, query: str, doc_score_pairs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Re-rank documents using cross-encoder if enabled."""
        if not self.config.use_reranking:
            return doc_score_pairs[:self.config.final_k]
        
        docs = [doc for doc, _ in doc_score_pairs]
        pairs = [[query, doc.content] for doc in docs]
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Normalize scores
        normalized_scores = self._normalize_scores(cross_scores)
        
        # Create new pairs with normalized scores
        reranked_pairs = list(zip(docs, normalized_scores))
        reranked_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_pairs[:self.config.final_k]

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve documents based on configured search mode."""
        if self.config.search_mode == SearchMode.SEMANTIC:
            indices, scores = self._semantic_search(query)
            initial_results = [(self.documents[idx], score) for idx, score in zip(indices, scores)]
        else:  # HYBRID
            initial_results = self._hybrid_search(query)
        
        return self._rerank(query, initial_results)

    def generate_response(self, query: str, retrieved_docs: List[Document]) -> str:
        """Generate response using GPT-4."""
        # Sort documents so system info appears first if present
        sorted_docs = sorted(retrieved_docs, key=lambda x: x.metadata.get('type') != 'system_info')
        context = "\n\n---\n\n".join(doc.content for doc in sorted_docs)
        
        prompt = f"""As Anura Aficionado, you are a specialized assistant for the Anura schema. Based on the following information:

{context}

Please answer this question: {query}

If this is a question about Anura Aficionado itself, answer based on the system information provided. If it's about the schema, use the schema information. If the needed information isn't in the context, please say so."""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Anura Aficionado, a specialized assistant that helps users understand and work with the Anura schema in Optilogic's Cosmic Frog platform."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content

    def query(self, query: str) -> Tuple[str, List[Tuple[Document, float]], float]:
        """Complete RAG pipeline with configured settings and timing."""
        start_time = time.time()
        retrieved_docs_with_scores = self.retrieve(query)
        response = self.generate_response(query, [doc for doc, _ in retrieved_docs_with_scores])
        execution_time = time.time() - start_time
        
        return response, retrieved_docs_with_scores, execution_time