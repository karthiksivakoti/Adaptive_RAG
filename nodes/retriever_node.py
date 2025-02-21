# risk_rag_system/nodes/retriever_node.py

from typing import List, Dict, Any, Optional
import torch
from loguru import logger
from pydantic import BaseModel
import numpy as np

from nodes.base_node import BaseNode, NodeInput, NodeOutput
from indexing.hybrid_embeddings.bge_embedder import BGEEmbedder
from indexing.hybrid_embeddings.splade_embedder import SpladeProcessor
from indexing.vectorstore import VectorStore

class RetrievalResult(BaseModel):
    """Structure for search results"""
    content: str
    score: float
    metadata: Dict[str, Any]

class RetrieverNodeInput(NodeInput):
    """Input structure for retriever node"""
    filters: Dict[str, Any] = {}
    top_k: int = 5

class RetrieverNodeOutput(NodeOutput):
    """Output structure for retriever node"""
    retrieved_documents: List[Dict[str, Any]]
    query: str

class RetrieverNode(BaseNode):
    """Node responsible for hybrid document retrieval"""
    
    async def _initialize_node(self) -> None:
        """Initialize retrieval components"""
        try:
            # Initialize embedders
            self.bge_embedder = BGEEmbedder()
            self.splade_processor = SpladeProcessor()
            self.min_confidence_threshold = 0.6
            
            # Initialize vector stores
            self.dense_store = VectorStore(
                store_type="chroma",
                collection_name="dense_vectors"
            )
            
            # Initialize test data
            await self._initialize_test_data()
            
            logger.info("Initialized RetrieverNode components")
        except Exception as e:
            logger.warning(f"Error initializing some components: {e}")
            self.splade_processor = None

    async def _initialize_test_data(self) -> None:
        """Initialize test data in vector store"""
        test_documents = [
            {
                "content": "Project risks include technical debt in legacy systems, resource constraints in Q3, and compliance requirements. Current status shows high impact on delivery timelines.",
                "metadata": {
                    "type": "risk",
                    "source": "analysis",
                    "priority": "high"
                }
            },
            {
                "content": "Risk mitigation strategies have been implemented: technical debt reduction program at 30% completion, new team onboarding completed, and initial compliance audit finished.",
                "metadata": {
                    "type": "risk",
                    "source": "update",
                    "priority": "medium"
                }
            },
            {
                "content": "Monthly risk assessment update: 1. Legacy system modernization in progress, 2. Resource allocation optimized for Q4, 3. Compliance framework established.",
                "metadata": {
                    "type": "risk",
                    "source": "assessment",
                    "priority": "medium"
                }
            }
        ]
        
        try:
            # Generate embeddings
            embeddings = await self.bge_embedder.encode(
                [doc["content"] for doc in test_documents]
            )
            
            # Add to vector store
            await self.dense_store.add(embeddings, test_documents)
            
            logger.info(f"Initialized {len(test_documents)} test documents")
        except Exception as e:
            logger.error(f"Error initializing test data: {e}")

    async def process(self, input_data: NodeInput) -> NodeOutput:
        """Process retrieval request"""
        logger.info("Processing retrieval request")
        
        try:
            # Convert to RetrieverNodeInput
            retriever_input = RetrieverNodeInput(
                content=input_data.content,
                filters=input_data.metadata.get("filters", {}),
                top_k=input_data.metadata.get("top_k", 5)
            )

            # Get dense and sparse retrievals
            dense_results = await self._dense_retrieval(retriever_input)
            sparse_results = await self._sparse_retrieval(retriever_input)
            
            # Combine results
            combined_results = self._hybrid_fusion(dense_results, sparse_results)
            
            # Calculate confidence score based on retrieval quality
            confidence_score = self._calculate_confidence(combined_results)
            
            return NodeOutput(
                content={
                    "results": [
                        {
                            "content": result.content,
                            "score": result.score,
                            "metadata": result.metadata
                        }
                        for result in combined_results
                    ]
                },
                confidence_score=confidence_score,
                metadata={
                    "num_results": len(combined_results),
                    "filters_applied": retriever_input.filters,
                    "retrieval_type": "hybrid",
                    "dense_results": len(dense_results or []),
                    "sparse_results": len(sparse_results or []),
                    "source_documents": [
                        {
                            "content": r.content,
                            "metadata": r.metadata
                        } for r in combined_results
                    ],
                    "retrieval_scores": {
                        "dense": max([r.score for r in dense_results]) if dense_results else 0.0,
                        "sparse": max([r.score for r in sparse_results]) if sparse_results else 0.0
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error in retrieval process: {e}")
            return NodeOutput(
                content={"results": []},
                confidence_score=0.0,
                metadata={"error": str(e)}
            )

    async def _dense_retrieval(
        self,
        input_data: RetrieverNodeInput
    ) -> List[RetrievalResult]:
        """Perform dense retrieval"""
        try:
            # Generate query embedding
            query_embedding = await self.bge_embedder.encode([input_data.content])
            
            # Search vector store
            results = await self.dense_store.search(
                query_embedding=query_embedding,
                k=input_data.top_k,
                filter_dict=input_data.filters
            )
            
            # Convert to RetrievalResults
            return [
                RetrievalResult(
                    content=r["content"],
                    score=r["score"],
                    metadata=r.get("metadata", {})
                )
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return []

    async def _sparse_retrieval(
        self,
        input_data: RetrieverNodeInput
    ) -> List[RetrievalResult]:
        """Perform sparse retrieval"""
        try:
            if not self.splade_processor:
                return []
                
            # Process query
            query_weights, _ = await self.splade_processor.process_text(
                input_data.content
            )
            
            # Search using sparse weights
            results = await self.dense_store.search(
                query_weights=query_weights,
                k=input_data.top_k,
                filter_dict=input_data.filters
            )
            
            # Convert to RetrievalResults
            return [
                RetrievalResult(
                    content=r["content"],
                    score=r["score"],
                    metadata=r.get("metadata", {})
                )
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {e}")
            return []

    def _hybrid_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Combine dense and sparse results"""
        try:
            # Handle empty results
            if not dense_results and not sparse_results:
                return [
                    RetrievalResult(
                        content="No matching documents found.",
                        score=0.1,
                        metadata={"source": "empty_result"}
                    )
                ]
                
            # Combine all results
            all_results = {}
            
            # Add dense results
            for rank, result in enumerate(dense_results or [], 1):
                if result.content not in all_results:
                    all_results[result.content] = {
                        "content": result.content,
                        "dense_score": result.score / (rank + 60),
                        "sparse_score": 0.0,
                        "metadata": result.metadata
                    }
                    
            # Add sparse results
            for rank, result in enumerate(sparse_results or [], 1):
                if result.content in all_results:
                    all_results[result.content]["sparse_score"] = result.score / (rank + 60)
                else:
                    all_results[result.content] = {
                        "content": result.content,
                        "dense_score": 0.0,
                        "sparse_score": result.score / (rank + 60),
                        "metadata": result.metadata
                    }

            # Calculate combined scores
            sorted_results = []
            for content, scores in all_results.items():
                combined_score = 0.7 * scores["dense_score"] + 0.3 * scores["sparse_score"]
                sorted_results.append(
                    RetrievalResult(
                        content=content,
                        score=combined_score,
                        metadata={
                            **scores["metadata"],
                            "dense_score": scores["dense_score"],
                            "sparse_score": scores["sparse_score"]
                        }
                    )
                )

            # Sort by combined score
            return sorted(
                sorted_results,
                key=lambda x: x.score,
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid fusion: {e}")
            return [
                RetrievalResult(
                    content="Error combining results",
                    score=0.0,
                    metadata={"error": str(e)}
                )
            ]

    def _calculate_confidence(self, results: List[RetrievalResult]) -> float:
        """Calculate confidence score for retrieval results"""
        try:
            if not results:
                return 0.0
                
            # Use top 3 scores for confidence
            top_scores = [r.score for r in results[:3]]
            avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
            
            # Apply sigmoid-like normalization
            confidence = 1 / (1 + np.exp(-10 * (avg_score - 0.5)))
            
            return float(max(min(confidence, 1.0), 0.0))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    async def route(self, output: NodeOutput) -> Optional[BaseNode]:
        """Route based on retrieval results"""
        results = output.content.get("results", [])
        
        if not results:
            logger.info("No results found, routing to web search")
            return self.next_nodes.get('no_results')
            
        if output.confidence_score < self.min_confidence_threshold:
            logger.info("Low confidence, routing to low confidence path")
            return self.next_nodes.get('low_confidence')
            
        logger.info("Routing to default path (grader)")
        return self.next_nodes.get('default')