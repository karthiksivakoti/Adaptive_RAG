# risk_rag_system/indexing/raptor/summary_indexer.py

from typing import List, Dict, Any, Optional, Tuple
import torch
from loguru import logger
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings

from .tree_builder import RaptorTree, RaptorTreeBuilder
from .recursive_summarizer import SummaryNode

class IndexerConfig(BaseModel):
    """Configuration for RAPTOR indexer"""
    collection_name: str = "raptor_summaries"
    distance_metric: str = "cosine"
    top_k: int = 5
    min_relevance_score: float = 0.6
    cache_embeddings: bool = True

class QueryResult(BaseModel):
    """Structure for query results"""
    content: str
    level: int
    relevance_score: float
    tree_id: str
    node_path: List[str]
    metadata: Dict[str, Any]

class RaptorIndexer:
    """Manages indexing and retrieval of RAPTOR summaries"""

    def __init__(
        self,
        config: Optional[IndexerConfig] = None,
        tree_builder: Optional[RaptorTreeBuilder] = None
    ):
        self.config = config or IndexerConfig()
        self.tree_builder = tree_builder or RaptorTreeBuilder()
        self._initialize_storage()
        logger.info("Initialized RAPTOR Indexer")

    def _initialize_storage(self) -> None:
        """Initialize vector storage"""
        try:
            self.client = chromadb.Client(Settings(
                persist_directory="./data/raptor_index"
            ))
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            logger.info("Initialized vector storage")
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            raise

    async def index_document(
        self,
        document: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> RaptorTree:
        """
        Index a document using RAPTOR
        
        Args:
            document: Document to index
            metadata: Optional metadata
            
        Returns:
            Built RAPTOR tree
        """
        try:
            # Build RAPTOR tree
            tree = await self.tree_builder.build_tree(document, metadata)
            
            # Index all nodes
            await self._index_tree(tree)
            
            logger.info(f"Successfully indexed document {document.get('id', '')}")
            return tree
            
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            raise

    async def _index_tree(self, tree: RaptorTree) -> None:
        """Index all nodes in a RAPTOR tree"""
        embeddings = []
        documents = []
        metadatas = []
        ids = []

        # Collect all nodes
        for node in self.tree_builder.traverse_tree(tree):
            node_id = str(id(node))
            
            embeddings.append(node.embedding.numpy().tolist())
            documents.append(node.content)
            metadatas.append({
                "level": node.level,
                "tree_id": tree.metadata["document_id"],
                **node.metadata
            })
            ids.append(node_id)

        # Batch add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    async def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        level_weights: Optional[Dict[int, float]] = None
    ) -> List[QueryResult]:
        """
        Query the indexed summaries
        
        Args:
            query: Query string
            filters: Optional metadata filters
            level_weights: Optional weights for different tree levels
            
        Returns:
            List of relevant results
        """
        try:
            # Get query embedding
            query_embedding = await self.tree_builder.embedder.encode([query])
            
            # Search collection
            results = self.collection.query(
                query_embeddings=query_embedding.numpy().tolist(),
                n_results=self.config.top_k,
                where=filters
            )

            # Process results
            query_results = []
            for i, (doc, metadata, score) in enumerate(zip(
                results["documents"],
                results["metadatas"],
                results["distances"]
            )):
                # Apply level weights if provided
                if level_weights and metadata["level"] in level_weights:
                    score *= level_weights[metadata["level"]]

                if score >= self.config.min_relevance_score:
                    result = QueryResult(
                        content=doc,
                        level=metadata["level"],
                        relevance_score=float(score),
                        tree_id=metadata["tree_id"],
                        node_path=self._get_node_path(metadata["tree_id"], doc),
                        metadata=metadata
                    )
                    query_results.append(result)

            return sorted(
                query_results,
                key=lambda x: x.relevance_score,
                reverse=True
            )

        except Exception as e:
            logger.error(f"Error querying index: {e}")
            raise

    def _get_node_path(self, tree_id: str, content: str) -> List[str]:
        """Get path from root to node in tree"""
        for tree in self.tree_builder.trees:
            if tree.metadata["document_id"] == tree_id:
                path = []
                current = tree.root
                while current:
                    path.append(current.content)
                    if current.content == content:
                        return path
                    # Find matching child
                    current = next(
                        (child for child in current.children
                         if content in child.content),
                        None
                    )
        return []

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the indexer"""
        return {
            "collection_name": self.config.collection_name,
            "num_documents": self.collection.count(),
            "config": self.config.dict()
        }