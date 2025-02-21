# risk_rag_system/indexing/vectorstore.py

from typing import Dict, Any, Optional, List, Tuple, Set
import torch
import numpy as np
from pydantic import BaseModel
from loguru import logger
import chromadb
from chromadb.config import Settings
import faiss
import json
from pathlib import Path
import asyncio
import time
from datetime import datetime
import pickle
from concurrent.futures import ThreadPoolExecutor

class VectorStoreConfig(BaseModel):
    """Configuration for vector store"""
    store_type: str = "chroma"  # chroma or faiss
    collection_name: str = "documents"
    distance_metric: str = "cosine"
    persist_directory: str = "./data/vectorstore"
    dimension: int = 768
    index_type: str = "hnsw"  # hnsw or flat
    max_elements: int = 1_000_000
    ef_construction: int = 200
    ef_search: int = 50
    nprobe: int = 10  # for IVF indexes
    cache_size: int = 100_000
    rebuild_threshold: int = 10_000
    thread_pool_size: int = 4

class VectorStore:
    """Vector store implementation with hybrid support"""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self._initialize_store()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.updates_since_rebuild = 0
        self._lock = asyncio.Lock()
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size
        )
        logger.info(f"Initialized VectorStore with type: {self.config.store_type}")

    def _initialize_store(self) -> None:
        """Initialize vector store backend"""
        try:
            if self.config.store_type == "chroma":
                self._initialize_chroma()
            elif self.config.store_type == "faiss":
                self._initialize_faiss()
            else:
                raise ValueError(f"Unknown store type: {self.config.store_type}")
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def _initialize_chroma(self) -> None:
        """Initialize ChromaDB"""
        persist_dir = Path(self.config.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.Client(Settings(
            persist_directory=str(persist_dir),
            chroma_db_impl="duckdb+parquet"
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={
                "hnsw:space": self.config.distance_metric,
                "hnsw:construction_ef": self.config.ef_construction,
                "hnsw:search_ef": self.config.ef_search
            }
        )

    def _initialize_faiss(self) -> None:
        """Initialize FAISS index"""
        # Create index based on configuration
        if self.config.index_type == "hnsw":
            # HNSW index for approximate search
            self.index = faiss.IndexHNSWFlat(
                self.config.dimension,
                self.config.ef_construction,
                faiss.METRIC_INNER_PRODUCT 
                if self.config.distance_metric == "cosine" 
                else faiss.METRIC_L2
            )
            self.index.hnsw.efSearch = self.config.ef_search
        else:
            # Flat index for exact search
            if self.config.distance_metric == "cosine":
                self.index = faiss.IndexFlatIP(self.config.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.config.dimension)
        
        # Load existing index if available
        index_path = Path(self.config.persist_directory) / "faiss_index.bin"
        metadata_path = Path(self.config.persist_directory) / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, 'rb') as f:
                self.metadata_store = pickle.load(f)
        else:
            self.metadata_store: Dict[int, Dict[str, Any]] = {}
    
    async def add(
        self,
        embeddings: torch.Tensor,
        documents: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> None:
        """Add documents to vector store"""
        async with self._lock:
            try:
                # Convert embeddings to numpy if needed
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()
                
                # Process in batches
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i + batch_size]
                    batch_embeddings = embeddings[i:i + batch_size]
                    
                    if self.config.store_type == "chroma":
                        await self._add_to_chroma(batch_embeddings, batch_docs)
                    else:
                        await self._add_to_faiss(batch_embeddings, batch_docs)
                    
                    # Update cache
                    for doc in batch_docs:
                        doc_id = doc.get("id", str(time.time()))
                        self.cache[doc_id] = doc
                        
                        # Limit cache size
                        if len(self.cache) > self.config.cache_size:
                            oldest_key = min(self.cache.keys())
                            del self.cache[oldest_key]
                    
                # Check if rebuild needed
                self.updates_since_rebuild += len(documents)
                if (
                    self.config.store_type == "faiss" and
                    self.updates_since_rebuild >= self.config.rebuild_threshold
                ):
                    await self._rebuild_index()
                
                logger.info(f"Added {len(documents)} documents to vector store")
                
            except Exception as e:
                logger.error(f"Error adding documents: {e}")
                raise

    async def _add_to_chroma(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]]
    ) -> None:
        """Add documents to ChromaDB"""
        # Prepare data for ChromaDB
        ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
        metadatas = [
            {k: v for k, v in doc.items() if k != "content"}
            for doc in documents
        ]
        texts = [doc["content"] for doc in documents]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    async def _add_to_faiss(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]]
    ) -> None:
        """Add documents to FAISS"""
        loop = asyncio.get_event_loop()
        
        # Run FAISS operations in thread pool
        await loop.run_in_executor(
            self.thread_pool,
            self._add_to_faiss_sync,
            embeddings,
            documents
        )

    def _add_to_faiss_sync(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]]
    ) -> None:
        """Synchronous FAISS addition"""
        if self.config.distance_metric == "cosine":
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        start_idx = len(self.metadata_store)
        for i, doc in enumerate(documents):
            self.metadata_store[start_idx + i] = doc
        
        # Save to disk
        self._save_faiss_index()

    def _save_faiss_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        persist_dir = Path(self.config.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = persist_dir / "faiss_index.bin"
        metadata_path = persist_dir / "metadata.pkl"
        
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)

    async def _rebuild_index(self) -> None:
        """Rebuild FAISS index for optimization"""
        if self.config.store_type != "faiss":
            return
            
        try:
            logger.info("Starting index rebuild")
            
            # Get all vectors and metadata
            total_vectors = self.index.ntotal
            all_vectors = np.zeros((total_vectors, self.config.dimension), dtype=np.float32)
            
            for i in range(total_vectors):
                vector = self.index.reconstruct(i)
                all_vectors[i] = vector
            
            # Create new optimized index
            if self.config.index_type == "hnsw":
                new_index = faiss.IndexHNSWFlat(
                    self.config.dimension,
                    self.config.ef_construction,
                    faiss.METRIC_INNER_PRODUCT 
                    if self.config.distance_metric == "cosine" 
                    else faiss.METRIC_L2
                )
                new_index.hnsw.efSearch = self.config.ef_search
            else:
                new_index = faiss.IndexFlatL2(self.config.dimension)
            
            # Add vectors to new index
            new_index.add(all_vectors)
            
            # Replace old index
            self.index = new_index
            self._save_faiss_index()
            
            self.updates_since_rebuild = 0
            logger.info("Completed index rebuild")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            raise

    async def search(
        self,
        query_embedding: torch.Tensor,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search vector store"""
        async with self._lock:
            try:
                # Convert query embedding
                if isinstance(query_embedding, torch.Tensor):
                    query_embedding = query_embedding.cpu().numpy()
                
                # Normalize for cosine similarity if needed
                if self.config.distance_metric == "cosine":
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                if self.config.store_type == "chroma":
                    results = await self._search_chroma(
                        query_embedding,
                        k,
                        filter_dict
                    )
                else:
                    results = await self._search_faiss(
                        query_embedding,
                        k,
                        filter_dict
                    )
                
                return results
                
            except Exception as e:
                logger.error(f"Error searching vector store: {e}")
                raise

    async def _search_chroma(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB"""
        # Execute search
        results = self.collection.query(
            query_embeddings=query_embedding.reshape(1, -1).tolist(),
            n_results=k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            formatted_results.append({
                "content": doc,
                "metadata": metadata,
                "score": 1 - distance  # Convert distance to similarity
            })
        
        return formatted_results

    async def _search_faiss(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search FAISS index"""
        loop = asyncio.get_event_loop()
        
        # Run search in thread pool
        distances, indices = await loop.run_in_executor(
            self.thread_pool,
            self._search_faiss_sync,
            query_embedding,
            k
        )
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # Invalid index
                continue
                
            doc = self.metadata_store.get(idx)
            if not doc:
                continue
                
            # Skip deleted documents
            if doc.get("deleted", False):
                continue
                
            # Apply filters if any
            if filter_dict and not self._matches_filter(doc, filter_dict):
                continue
                
            results.append({
                "content": doc.get("content", ""),
                "metadata": {k: v for k, v in doc.items() if k != "content"},
                "score": float(1 - distances[0][i] if self.config.distance_metric == "cosine"
                             else -distances[0][i])  # Convert distance to similarity
            })
        
        return results

    def _search_faiss_sync(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronous FAISS search"""
        return self.index.search(
            query_embedding.reshape(1, -1),
            k
        )

    def _matches_filter(
        self,
        doc: Dict[str, Any],
        filter_dict: Dict[str, Any]
    ) -> bool:
        """Check if document matches filter criteria"""
        for key, value in filter_dict.items():
            if key not in doc:
                return False
            if isinstance(value, list):
                if doc[key] not in value:
                    return False
            elif doc[key] != value:
                return False
        return True

    async def delete(
        self,
        document_ids: List[str]
    ) -> None:
        """Delete documents from vector store"""
        async with self._lock:
            try:
                if self.config.store_type == "chroma":
                    self.collection.delete(ids=document_ids)
                else:
                    # FAISS doesn't support deletion, mark as deleted in metadata
                    for doc_id in document_ids:
                        for idx, metadata in self.metadata_store.items():
                            if metadata.get("id") == doc_id:
                                metadata["deleted"] = True
                                self._save_faiss_index()
                
                # Remove from cache
                for doc_id in document_ids:
                    self.cache.pop(doc_id, None)
                
                logger.info(f"Deleted {len(document_ids)} documents")
                
            except Exception as e:
                logger.error(f"Error deleting documents: {e}")
                raise

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            if self.config.store_type == "chroma":
                count = self.collection.count()
            else:
                count = self.index.ntotal
            
            return {
                "total_documents": count,
                "cache_size": len(self.cache),
                "updates_since_rebuild": self.updates_since_rebuild,
                "store_type": self.config.store_type,
                "index_type": self.config.index_type
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Persist data
            if self.config.store_type == "chroma":
                self.client.persist()
            else:
                self._save_faiss_index()
            
            # Clear cache
            self.cache.clear()
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Vector store cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise

    def get_state(self) -> Dict[str, Any]:
        """Get current state of vector store"""
        return {
            "store_type": self.config.store_type,
            "dimension": self.config.dimension,
            "index_type": self.config.index_type,
            "distance_metric": self.config.distance_metric,
            "cache_size": len(self.cache),
            "persist_directory": str(self.config.persist_directory)
        }            