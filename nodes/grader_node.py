# risk_rag_system/nodes/grader_node.py

from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from loguru import logger

from nodes.base_node import BaseNode, NodeInput, NodeOutput
from llm.mistral_router import MistralRouter

class GraderMetrics(BaseModel):
    """Metrics for grading retrieved content"""
    relevance_score: float
    hallucination_score: float
    factual_consistency: float
    context_coverage: float

class GraderNodeInput(NodeInput):
    """Input structure for grader node"""
    query: str
    retrieved_documents: List[Dict[str, Any]]
    context: str = ""
    
class GraderNodeOutput(NodeOutput):
    """Output structure for grader node"""
    graded_documents: List[Dict[str, Any]]
    metrics: GraderMetrics
    requires_rewrite: bool = False
    rewrite_reason: str = ""

class GraderNode(BaseNode):
    """Node responsible for grading retrieved content"""
    
    def _initialize_node(self) -> None:
        """Initialize grading components"""
        self.model = MistralRouter()
        self.relevance_threshold = 0.7
        self.hallucination_threshold = 0.2
        self.min_context_coverage = 0.6
        logger.info("Initialized GraderNode components")

    async def process(self, input_data: GraderNodeInput) -> GraderNodeOutput:
        """Process and grade retrieved content"""
        logger.info(f"Grading content for query: {input_data.query}")
        
        try:
            # Grade each document
            graded_docs = []
            total_relevance = 0
            total_hallucination = 0
            
            for doc in input_data.retrieved_documents:
                relevance, hallucination = await self._grade_document(
                    query=input_data.query,
                    document=doc,
                    context=input_data.context
                )
                
                graded_doc = {
                    **doc,
                    "relevance_score": relevance,
                    "hallucination_score": hallucination
                }
                graded_docs.append(graded_doc)
                total_relevance += relevance
                total_hallucination += hallucination

            # Calculate average metrics
            avg_relevance = total_relevance / len(input_data.retrieved_documents)
            avg_hallucination = total_hallucination / len(input_data.retrieved_documents)
            
            # Calculate context coverage
            context_coverage = self._calculate_context_coverage(
                query=input_data.query,
                documents=graded_docs
            )
            
            # Check if query rewrite is needed
            requires_rewrite, rewrite_reason = self._check_rewrite_needed(
                avg_relevance, 
                avg_hallucination,
                context_coverage
            )

            metrics = GraderMetrics(
                relevance_score=avg_relevance,
                hallucination_score=avg_hallucination,
                factual_consistency=1 - avg_hallucination,
                context_coverage=context_coverage
            )

            # Calculate overall confidence
            confidence_score = self._calculate_confidence(metrics)

            return GraderNodeOutput(
                content=graded_docs,
                confidence_score=confidence_score,
                graded_documents=graded_docs,
                metrics=metrics,
                requires_rewrite=requires_rewrite,
                rewrite_reason=rewrite_reason,
                metadata={
                    "num_documents_graded": len(graded_docs),
                    "original_query": input_data.query
                }
            )

        except Exception as e:
            logger.error(f"Error in grading process: {e}")
            raise

    async def _grade_document(
        self, 
        query: str, 
        document: Dict[str, Any],
        context: str
    ) -> Tuple[float, float]:
        """Grade a single document for relevance and hallucination"""
        
        # Construct prompt for relevance checking
        relevance_prompt = f"""Grade the relevance of this document to the query.
        Query: {query}
        Document: {document['content']}
        
        Return a score between 0 and 1, where:
        1 = Perfectly relevant
        0 = Completely irrelevant
        
        Consider: topic match, information completeness, and answer coverage.
        """
        
        # Construct prompt for hallucination checking
        hallucination_prompt = f"""Check this content for potential hallucinations.
        Content: {document['content']}
        Context: {context}
        
        Return a score between 0 and 1, where:
        1 = Completely hallucinated
        0 = Fully grounded in context
        
        Focus on: factual consistency, unsupported claims, and verifiability.
        """
        
        # Get grades from model
        relevance_score = float(await self.model.get_rating(relevance_prompt))
        hallucination_score = float(await self.model.get_rating(hallucination_prompt))
        
        return relevance_score, hallucination_score

    def _calculate_context_coverage(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well the documents cover the query requirements"""
        # Implement context coverage calculation
        # This could use techniques like:
        # - Query term coverage
        # - Semantic similarity
        # - Required information presence
        return 0.8  # Placeholder

    def _check_rewrite_needed(
        self,
        relevance: float,
        hallucination: float,
        coverage: float
    ) -> Tuple[bool, str]:
        """Determine if query needs to be rewritten"""
        if relevance < self.relevance_threshold:
            return True, "Low relevance in retrieved documents"
        if hallucination > self.hallucination_threshold:
            return True, "High hallucination risk detected"
        if coverage < self.min_context_coverage:
            return True, "Insufficient context coverage"
        return False, ""

    def _calculate_confidence(self, metrics: GraderMetrics) -> float:
        """Calculate overall confidence score"""
        weights = {
            'relevance': 0.4,
            'hallucination': 0.3,
            'coverage': 0.3
        }
        
        confidence = (
            weights['relevance'] * metrics.relevance_score +
            weights['hallucination'] * (1 - metrics.hallucination_score) +
            weights['coverage'] * metrics.context_coverage
        )
        
        return max(min(confidence, 1.0), 0.0)

    async def route(self, output: GraderNodeOutput) -> Optional[BaseNode]:
        """Route based on grading results"""
        if output.requires_rewrite:
            return self.next_nodes.get('rewrite')
        if output.confidence_score < 0.5:
            return self.next_nodes.get('low_confidence')
        return self.next_nodes.get('default')