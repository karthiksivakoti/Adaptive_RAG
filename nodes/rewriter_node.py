# risk_rag_system/nodes/rewriter_node.py

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from loguru import logger

from nodes.base_node import BaseNode, NodeInput, NodeOutput
from llm.mistral_router import MistralRouter

class QueryDecomposition(BaseModel):
    """Structure for decomposed query parts"""
    sub_queries: List[str]
    dependencies: Dict[str, List[str]]
    reasoning: str

class RewriterNodeInput(NodeInput):
    """Input structure for rewriter node"""
    original_query: str
    grading_metrics: Dict[str, float]
    rewrite_reason: str
    failed_attempts: List[str] = []
    context: str = ""

class RewriterNodeOutput(NodeOutput):
    """Output structure for rewriter node"""
    rewritten_query: str
    decomposition: Optional[QueryDecomposition] = None
    requires_web_search: bool = False

class RewriterNode(BaseNode):
    """Node responsible for query rewriting and decomposition"""
    
    def _initialize_node(self) -> None:
        """Initialize rewriter components"""
        self.model = MistralRouter()
        self.max_rewrite_attempts = 3
        self.web_search_threshold = 0.4
        logger.info("Initialized RewriterNode components")

    async def process(self, input_data: RewriterNodeInput) -> RewriterNodeOutput:
        """Process and rewrite query"""
        logger.info(f"Rewriting query: {input_data.original_query}")
        
        try:
            # Check if we should try web search
            if len(input_data.failed_attempts) >= self.max_rewrite_attempts:
                return RewriterNodeOutput(
                    content=input_data.original_query,
                    confidence_score=self.web_search_threshold,
                    rewritten_query=input_data.original_query,
                    requires_web_search=True,
                    metadata={
                        "reason": "Max rewrite attempts reached",
                        "failed_attempts": input_data.failed_attempts
                    }
                )

            # Analyze query for decomposition
            should_decompose = await self._should_decompose(input_data.original_query)
            
            if should_decompose:
                # Decompose query into sub-queries
                decomposition = await self._decompose_query(
                    query=input_data.original_query,
                    context=input_data.context
                )
                
                # Use decomposition to generate better query
                rewritten_query = await self._generate_from_decomposition(decomposition)
            else:
                # Direct rewrite
                rewritten_query = await self._rewrite_query(
                    query=input_data.original_query,
                    reason=input_data.rewrite_reason,
                    context=input_data.context,
                    failed_attempts=input_data.failed_attempts
                )
                decomposition = None

            # Calculate confidence in rewrite
            confidence_score = await self._calculate_rewrite_confidence(
                original_query=input_data.original_query,
                rewritten_query=rewritten_query,
                grading_metrics=input_data.grading_metrics
            )

            return RewriterNodeOutput(
                content=rewritten_query,
                confidence_score=confidence_score,
                rewritten_query=rewritten_query,
                decomposition=decomposition,
                requires_web_search=confidence_score < self.web_search_threshold,
                metadata={
                    "original_query": input_data.original_query,
                    "rewrite_reason": input_data.rewrite_reason,
                    "attempt_number": len(input_data.failed_attempts) + 1
                }
            )

        except Exception as e:
            logger.error(f"Error in rewriting process: {e}")
            raise

    async def _should_decompose(self, query: str) -> bool:
        """Determine if query needs decomposition"""
        prompt = f"""Analyze if this query needs to be decomposed into sub-queries.
        Query: {query}
        
        Consider:
        1. Query complexity
        2. Multiple distinct aspects
        3. Dependent information needs
        
        Return 'True' or 'False'
        """
        
        response = await self.model.get_completion(prompt)
        return response.strip().lower() == 'true'

    async def _decompose_query(
        self, 
        query: str,
        context: str
    ) -> QueryDecomposition:
        """Decompose complex query into sub-queries"""
        prompt = f"""Decompose this query into sub-queries:
        Query: {query}
        Context: {context}
        
        For each sub-query:
        1. Ensure it's self-contained
        2. Identify dependencies between sub-queries
        3. Provide reasoning for decomposition
        
        Format: JSON with sub_queries, dependencies, and reasoning
        """
        
        response = await self.model.get_structured_output(prompt)
        return QueryDecomposition(**response)

    async def _generate_from_decomposition(
        self, 
        decomposition: QueryDecomposition
    ) -> str:
        """Generate optimized query from decomposition"""
        prompt = f"""Create an optimized single query from these sub-queries:
        Sub-queries: {decomposition.sub_queries}
        Dependencies: {decomposition.dependencies}
        
        Requirements:
        1. Capture all information needs
        2. Maintain logical flow
        3. Clear and concise
        """
        
        return await self.model.get_completion(prompt)

    async def _rewrite_query(
        self,
        query: str,
        reason: str,
        context: str,
        failed_attempts: List[str]
    ) -> str:
        """Rewrite query based on previous failures"""
        prompt = f"""Rewrite this query to address the following issue:
        Original Query: {query}
        Issue: {reason}
        Previous Attempts: {failed_attempts}
        Context: {context}
        
        Requirements:
        1. Address the specific issue
        2. Maintain original intent
        3. Be more specific/clear
        4. Avoid patterns from failed attempts
        """
        
        return await self.model.get_completion(prompt)

    async def _calculate_rewrite_confidence(
        self,
        original_query: str,
        rewritten_query: str,
        grading_metrics: Dict[str, float]
    ) -> float:
        """Calculate confidence in the rewrite"""
        prompt = f"""Rate the quality of this query rewrite:
        Original: {original_query}
        Rewritten: {rewritten_query}
        Previous Metrics: {grading_metrics}
        
        Return a score between 0 and 1, considering:
        1. Intent preservation
        2. Clarity improvement
        3. Specificity
        """
        
        confidence = float(await self.model.get_rating(prompt))
        return max(min(confidence, 1.0), 0.0)

    async def route(self, output: RewriterNodeOutput) -> Optional[BaseNode]:
        """Route based on rewriting results"""
        if output.requires_web_search:
            return self.next_nodes.get('web_search')
        return self.next_nodes.get('default')