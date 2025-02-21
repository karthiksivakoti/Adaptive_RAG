# risk_rag_system/retrieval/hybrid_search.py

from typing import List, Dict, Any, Optional
import torch
from loguru import logger
from pydantic import BaseModel

from indexing.hybrid_embeddings.bge_embedder import BGEEmbedder
from indexing.hybrid_embeddings.splade_embedder import SpladeProcessor

class SearchResult(BaseModel):
    """Structure for search results"""
    document_id: str
    content: str
    dense_score: float
    sparse_score: float
    combined_score: float
    metadata: Dict[str, Any]

class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search"""
    dense_weight: float = 0.5  # Weight for dense scores
    sparse_weight: float = 0.5  # Weight for sparse scores
    top_k: int = 10  # Number of results to return
    min_score: float = 0.1  # Minimum score threshold

class HybridSearch:
    """Combines dense and sparse search for improved retrieval"""
    
    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None
    ):
        self.config = config or HybridSearchConfig()
        self.bge_embedder = BGEEmbedder()
        self.splade_processor = SpladeProcessor()
        self.documents = []
        self.dense_embeddings = None
        self.sparse_weights = []
        logger.info("Initialized Hybrid Search")

    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "content"
    ) -> None:
        """
        Index documents for both dense and sparse retrieval
        
        Args:
            documents: List of documents to index
            text_key: Key for text content in documents
        """
        try:
            # Generate dense embeddings
            logger.info("Generating dense embeddings...")
            dense_embeddings = await self.bge_embedder.encode_documents(
                documents,
                text_key
            )
            
            # Generate sparse representations
            logger.info("Generating sparse representations...")
            sparse_weights = await self.splade_processor.process_documents(
                documents,
                text_key
            )
            
            # Store the representations
            self.documents = documents
            self.dense_embeddings = dense_embeddings
            self.sparse_weights = sparse_weights
            
            logger.info(f"Indexed {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search
        
        Args:
            query: Search query
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        try:
            # Get query representations
            dense_query = await self.bge_embedder.encode_query(query)
            sparse_query, _ = await self.splade_processor.process_text(query)
            
            # Compute dense scores
            dense_scores = self._compute_dense_scores(dense_query)
            
            # Compute sparse scores
            sparse_scores = await self._compute_sparse_scores(sparse_query)
            
            # Combine scores
            combined_scores = self._combine_scores(dense_scores, sparse_scores)
            
            # Get top-k results
            results = await self._get_top_results(
                combined_scores,
                dense_scores,
                sparse_scores,
                filters
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise

    def _compute_dense_scores(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Compute dense similarity scores"""
        # Normalize query embedding
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
        
        # Compute cosine similarity
        scores = torch.matmul(
            self.dense_embeddings,
            query_embedding.unsqueeze(1)
        ).squeeze()
        
        return scores

    async def _compute_sparse_scores(
        self,
        query_weights: Dict[str, float]
    ) -> torch.Tensor:
        """Compute sparse similarity scores"""
        scores = []
        for doc_weights in self.sparse_weights:
            score = await self.splade_processor.compute_similarity(
                query_weights,
                doc_weights
            )
            scores.append(score)
        return torch.tensor(scores)

    def _combine_scores(
        self,
        dense_scores: torch.Tensor,
        sparse_scores: torch.Tensor
    ) -> torch.Tensor:
        """Combine dense and sparse scores"""
        # Normalize scores to [0, 1] range
        dense_scores = (dense_scores - dense_scores.min()) / (
            dense_scores.max() - dense_scores.min() + 1e-6
        )
        sparse_scores = (sparse_scores - sparse_scores.min()) / (
            sparse_scores.max() - sparse_scores.min() + 1e-6
        )
        
        # Weighted combination
        combined_scores = (
            self.config.dense_weight * dense_scores +
            self.config.sparse_weight * sparse_scores
        )
        
        return combined_scores

    async def _get_top_results(
        self,
        combined_scores: torch.Tensor,
        dense_scores: torch.Tensor,
        sparse_scores: torch.Tensor,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Get top-k results with optional filtering"""
        # Apply score threshold
        mask = combined_scores >= self.config.min_score
        
        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                filter_mask = torch.tensor([
                    doc.get(key) == value for doc in self.documents
                ])
                mask = mask & filter_mask
        
        # Get indices of valid documents
        valid_indices = torch.where(mask)[0]
        
        # Sort by combined score
        sorted_indices = valid_indices[
            torch.argsort(combined_scores[valid_indices], descending=True)
        ]
        
        # Take top-k
        top_k_indices = sorted_indices[:self.config.top_k]
        
        # Create results
        results = []
        for idx in top_k_indices:
            idx = idx.item()
            document = self.documents[idx]
            
            result = SearchResult(
                document_id=str(idx),
                content=document["content"],
                dense_score=float(dense_scores[idx]),
                sparse_score=float(sparse_scores[idx]),
                combined_score=float(combined_scores[idx]),
                metadata={
                    k: v for k, v in document.items()
                    if k != "content"
                }
            )
            results.append(result)
        
        return results

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the search system"""
        return {
            "num_documents": len(self.documents),
            "dense_embeddings_shape": (
                tuple(self.dense_embeddings.shape)
                if self.dense_embeddings is not None
                else None
            ),
            "num_sparse_weights": len(self.sparse_weights),
            "config": self.config.dict()
        }