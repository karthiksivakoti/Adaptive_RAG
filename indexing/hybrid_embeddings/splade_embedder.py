# risk_rag_system/indexing/hybrid_embeddings/splade_processor.py

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from loguru import logger
from pydantic import BaseModel
from collections import defaultdict

class SpladeConfig(BaseModel):
    """Configuration for SPLADE processing"""
    model_name: str = "facebook/contriever-msmarco"
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_k_terms: int = 256  # Maximum number of terms to keep per document
    threshold: float = 0.01  # Minimum activation threshold

class SpladeProcessor:
    """Handler for SPLADE-based sparse representations"""
    
    def __init__(self, config: Optional[SpladeConfig] = None):
        self.config = config or SpladeConfig()
        self._initialize_model()
        logger.info(f"Initialized SPLADE Processor with model: {self.config.model_name}")

    def _initialize_model(self) -> None:
        """Initialize the SPLADE model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
            self.model.to(self.config.device)
            self.model.eval()
            
            # Cache vocabulary for term mapping
            self.vocab = {
                idx: token for token, idx in self.tokenizer.vocab.items()
            }
            
            logger.info("Successfully loaded SPLADE model and tokenizer")
        except Exception as e:
            logger.error(f"Error initializing SPLADE model: {e}. Using fallback sparse encoding.")
            # Initialize basic fallback tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = None
            self.vocab = {
                idx: token for token, idx in self.tokenizer.vocab.items()
            }

    @torch.no_grad()
    async def process_text(
        self, 
        text: str
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Process text to generate SPLADE representation
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple of (term weights dictionary, sparse vector)
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.config.device)
            
            # Generate SPLADE activations
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply ReLU and max pooling
            activations = torch.max(
                F.relu(logits[0]), 
                dim=0
            )[0]
            
            # Get top-k terms
            top_k_values, top_k_indices = torch.topk(
                activations,
                k=min(self.config.top_k_terms, len(activations)),
                largest=True
            )
            
            # Create sparse representation
            term_weights = {}
            for value, idx in zip(
                top_k_values.cpu().tolist(),
                top_k_indices.cpu().tolist()
            ):
                if value >= self.config.threshold:
                    term = self.vocab[idx]
                    if not term.startswith("##"):  # Skip subword tokens
                        term_weights[term] = value

            # Create sparse vector
            sparse_vector = torch.zeros_like(activations)
            sparse_vector[top_k_indices] = top_k_values

            return term_weights, sparse_vector.cpu()

        except Exception as e:
            logger.error(f"Error processing text with SPLADE: {e}")
            raise

    async def process_batch(
        self, 
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """
        Process a batch of texts
        
        Args:
            texts: List of texts to process
            batch_size: Batch size for processing
            
        Returns:
            List of term weight dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                weights, _ = await self.process_text(text)
                batch_results.append(weights)
            
            results.extend(batch_results)
        
        return results

    async def process_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "content"
    ) -> List[Dict[str, float]]:
        """
        Process a list of documents
        
        Args:
            documents: List of document dictionaries
            text_key: Key for text content in documents
            
        Returns:
            List of term weight dictionaries
        """
        texts = [doc[text_key] for doc in documents]
        return await self.process_batch(texts)

    async def compute_similarity(
        self,
        query_weights: Dict[str, float],
        doc_weights: Dict[str, float]
    ) -> float:
        """
        Compute similarity between query and document weights
        
        Args:
            query_weights: Query term weights
            doc_weights: Document term weights
            
        Returns:
            Similarity score
        """
        score = 0.0
        for term, q_weight in query_weights.items():
            if term in doc_weights:
                score += q_weight * doc_weights[term]
        return score

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the processor"""
        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "vocab_size": len(self.vocab),
            "top_k_terms": self.config.top_k_terms
        }