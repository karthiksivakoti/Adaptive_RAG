# risk_rag_system/indexing/hybrid_embeddings/bge_embedder.py

from typing import List, Dict, Any, Optional
import torch
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from pydantic import BaseModel

class EmbeddingConfig(BaseModel):
    """Configuration for BGE embeddings"""
    model_name: str = "BAAI/bge-large-en-v1.5"
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_embeddings: bool = True
    instruction: str = "Represent this sentence for retrieval: "

class BGEEmbedder:
    """Handler for BGE embeddings generation"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._initialize_model()
        logger.info(f"Initialized BGE Embedder with model: {self.config.model_name}")

    def _initialize_model(self) -> None:
        """Initialize the BGE model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self.model.to(self.config.device)
            self.model.eval()
            logger.info("Successfully loaded BGE model and tokenizer")
        except Exception as e:
            logger.error(f"Error initializing BGE model: {e}")
            raise

    @torch.no_grad()
    async def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            Tensor of embeddings
        """
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Add instruction to each text
                processed_texts = [
                    f"{self.config.instruction}{text}" for text in batch_texts
                ]
                
                # Tokenize
                encoded_input = self.tokenizer(
                    processed_texts,
                    max_length=self.config.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded_input = {
                    k: v.to(self.config.device) for k, v in encoded_input.items()
                }
                
                # Generate embeddings
                outputs = self.model(**encoded_input)
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state,
                    encoded_input['attention_mask']
                )
                
                if self.config.normalize_embeddings:
                    embeddings = normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())

            # Concatenate all batches
            final_embeddings = torch.cat(all_embeddings, dim=0)
            logger.debug(f"Generated embeddings with shape: {final_embeddings.shape}")
            
            return final_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def _mean_pooling(
        self, 
        token_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings
        
        Args:
            token_embeddings: Token-level embeddings
            attention_mask: Attention mask for valid tokens
            
        Returns:
            Pooled embeddings
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    async def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a single query
        
        Args:
            query: Query text to encode
            
        Returns:
            Query embedding tensor
        """
        embeddings = await self.encode([query])
        return embeddings[0]

    async def encode_documents(
        self, 
        documents: List[Dict[str, Any]], 
        text_key: str = "content"
    ) -> torch.Tensor:
        """
        Encode a list of documents
        
        Args:
            documents: List of document dictionaries
            text_key: Key for text content in documents
            
        Returns:
            Document embeddings tensor
        """
        texts = [doc[text_key] for doc in documents]
        return await self.encode(texts)

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the embedder"""
        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "max_length": self.config.max_length
        }