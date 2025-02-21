# risk_rag_system/validation/confidence_scoring/scorer.py

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import numpy as np
from loguru import logger
import torch
from dataclasses import dataclass

class ScoringMetrics(BaseModel):
    """Metrics for confidence scoring"""
    relevance_score: float
    source_quality: float
    answer_completeness: float
    context_coverage: float
    factual_consistency: float
    semantic_similarity: float
    retrieval_confidence: float
    generation_confidence: float

class ScoringConfig(BaseModel):
    """Configuration for confidence scoring"""
    weights: Dict[str, float] = {
        "relevance_score": 0.2,
        "source_quality": 0.15,
        "answer_completeness": 0.15,
        "context_coverage": 0.1,
        "factual_consistency": 0.15,
        "semantic_similarity": 0.1,
        "retrieval_confidence": 0.1,
        "generation_confidence": 0.05
    }
    min_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    use_dynamic_weights: bool = True

@dataclass
class ConfidenceScore:
    """Structure for confidence score"""
    score: float
    metrics: ScoringMetrics
    reasoning: Dict[str, Any]
    threshold: float
    is_high_confidence: bool

class ConfidenceScorer:
    """Calculates confidence scores for generated responses"""
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        self._validate_weights()
        logger.info("Initialized ConfidenceScorer")

    def _validate_weights(self) -> None:
        """Validate weight configuration"""
        total = sum(self.config.weights.values())
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    async def calculate_confidence(
        self,
        response: Dict[str, Any],
        query: str,
        context: List[Dict[str, Any]],
        retrieval_scores: Dict[str, float],
        generation_metadata: Dict[str, Any]
    ) -> ConfidenceScore:
        """Calculate overall confidence score"""
        try:
            # Extract the actual content from response dictionary
            response_content = response.get("content", "")
            if isinstance(response_content, dict):
                response_content = response_content.get("results", [])
                # Convert list of results to a single string if needed
                if isinstance(response_content, list):
                    response_content = " ".join([r.get("content", "") for r in response_content])
            
            # Calculate individual metrics
            metrics = await self._calculate_metrics(
                response_content,
                query,
                context,
                retrieval_scores,
                generation_metadata
            )
            
            # Calculate weighted score
            weights = await self._get_dynamic_weights(metrics) if self.config.use_dynamic_weights else self.config.weights
            
            score = sum(
                getattr(metrics, metric) * weight
                for metric, weight in weights.items()
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(metrics, weights, score)
            
            # Determine confidence level
            is_high_confidence = score >= self.config.high_confidence_threshold
            
            return ConfidenceScore(
                score=score,
                metrics=metrics,
                reasoning=reasoning,
                threshold=self.config.min_threshold,
                is_high_confidence=is_high_confidence
            )
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            # Return a default confidence score on error
            default_metrics = ScoringMetrics(
                relevance_score=0.5,
                source_quality=0.5,
                answer_completeness=0.5,
                context_coverage=0.5,
                factual_consistency=0.5,
                semantic_similarity=0.5,
                retrieval_confidence=0.5,
                generation_confidence=0.5
            )
            return ConfidenceScore(
                score=0.5,
                metrics=default_metrics,
                reasoning={"error": str(e)},
                threshold=self.config.min_threshold,
                is_high_confidence=False
            )
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            raise

    async def _calculate_metrics(
        self,
        response: str,
        query: str,
        context: List[Dict[str, Any]],
        retrieval_scores: Dict[str, float],
        generation_metadata: Dict[str, Any]
    ) -> ScoringMetrics:
        """Calculate individual confidence metrics"""
        try:
            # Calculate relevance
            relevance_score = await self._calculate_relevance(
                response,
                query,
                context
            )
            
            # Evaluate source quality
            source_quality = self._evaluate_sources(context)
            
            # Check answer completeness
            answer_completeness = await self._check_completeness(
                response,
                query,
                context
            )
            
            # Calculate context coverage
            context_coverage = self._calculate_coverage(
                response,
                context
            )
            
            # Check factual consistency
            factual_consistency = await self._check_consistency(
                response,
                context
            )
            
            # Calculate semantic similarity
            semantic_similarity = await self._calculate_similarity(
                response,
                query,
                context
            )
            
            # Get confidence from retrieval and generation
            retrieval_confidence = ConfidenceScorer.safe_mean(list(retrieval_scores.values())) if retrieval_scores else 0.5
            generation_confidence = generation_metadata.get("confidence", 0.5)
            
            return ScoringMetrics(
                relevance_score=relevance_score,
                source_quality=source_quality,
                answer_completeness=answer_completeness,
                context_coverage=context_coverage,
                factual_consistency=factual_consistency,
                semantic_similarity=semantic_similarity,
                retrieval_confidence=retrieval_confidence,
                generation_confidence=generation_confidence
            )
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return default metrics on error
            return ScoringMetrics(
                relevance_score=0.5,
                source_quality=0.5,
                answer_completeness=0.5,
                context_coverage=0.5,
                factual_consistency=0.5,
                semantic_similarity=0.5,
                retrieval_confidence=0.5,
                generation_confidence=0.5
            )

    async def _calculate_relevance(
        self,
        response: str,
        query: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Calculate relevance score"""
        # Calculate semantic similarity between response and query
        query_similarity = await self._semantic_similarity(response, query)
        
        # Calculate relevance to context
        context_relevance = np.mean([
            await self._semantic_similarity(response, doc["content"])
            for doc in context
        ])
        
        # Combine scores
        return 0.6 * query_similarity + 0.4 * context_relevance

    def _evaluate_sources(
        self,
        context: List[Dict[str, Any]]
    ) -> float:
        """Evaluate quality of source documents"""
        scores = []
        for doc in context:
            # Check metadata for quality indicators
            metadata = doc.get("metadata", {})
            
            # Calculate source score based on various factors
            source_score = 0.0
            if metadata.get("verified", False):
                source_score += 0.3
            if metadata.get("source_type") == "official":
                source_score += 0.3
            if metadata.get("last_updated"):
                # Score based on recency
                score_age = min(1.0, 1 / (1 + metadata["age_days"]/365))
                source_score += 0.4 * score_age
                
            scores.append(min(1.0, source_score))
        
        return np.mean(scores) if scores else 0.0

    async def _check_completeness(
        self,
        response: str,
        query: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Check if response completely answers the query"""
        # Extract key elements from query
        query_elements = await self._extract_query_elements(query)
        
        # Check coverage of each element
        coverage_scores = []
        for element in query_elements:
            # Check if element is addressed in response
            element_score = await self._semantic_similarity(
                element,
                response
            )
            coverage_scores.append(element_score)
        
        return np.mean(coverage_scores) if coverage_scores else 0.0

    def _calculate_coverage(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well response covers context"""
        # Extract key information from context
        context_info = set()
        for doc in context:
            # Extract key phrases and entities
            info = self._extract_key_info(doc["content"])
            context_info.update(info)
        
        # Extract key information from response
        response_info = self._extract_key_info(response)
        
        # Calculate coverage
        if not context_info:
            return 0.0
            
        return len(response_info.intersection(context_info)) / len(context_info)

    async def _check_consistency(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Check factual consistency with context"""
        # Extract facts from response
        response_facts = await self._extract_facts(response)
        
        # Extract facts from context
        context_facts = []
        for doc in context:
            facts = await self._extract_facts(doc["content"])
            context_facts.extend(facts)
        
        # Compare facts
        consistency_scores = []
        for r_fact in response_facts:
            max_consistency = 0.0
            for c_fact in context_facts:
                consistency = await self._fact_similarity(r_fact, c_fact)
                max_consistency = max(max_consistency, consistency)
            consistency_scores.append(max_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0

    async def _calculate_similarity(
        self,
        response: str,
        query: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Calculate semantic similarity"""
        # Calculate similarity with query
        query_sim = await self._semantic_similarity(response, query)
        
        # Calculate similarity with context
        context_sims = []
        for doc in context:
            sim = await self._semantic_similarity(
                response,
                doc["content"]
            )
            context_sims.append(sim)
        
        context_sim = np.mean(context_sims) if context_sims else 0.0
        
        # Combine scores
        return 0.4 * query_sim + 0.6 * context_sim

    async def _get_dynamic_weights(
        self,
        metrics: ScoringMetrics
    ) -> Dict[str, float]:
        """Calculate dynamic weights based on metrics"""
        weights = self.config.weights.copy()
        
        # Adjust weights based on metric values
        if metrics.retrieval_confidence < 0.5:
            weights["context_coverage"] *= 1.2
            weights["factual_consistency"] *= 1.2
        
        if metrics.source_quality < 0.5:
            weights["factual_consistency"] *= 1.3
            
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _generate_reasoning(
        self,
        metrics: ScoringMetrics,
        weights: Dict[str, float],
        final_score: float
    ) -> Dict[str, Any]:
        """Generate reasoning for confidence score"""
        reasoning = {
            "final_score": final_score,
            "metric_scores": metrics.dict(),
            "weights_used": weights,
            "score_breakdown": {
                metric: (getattr(metrics, metric) * weights[metric])
                for metric in weights
            },
            "confidence_level": "high" if final_score >= self.config.high_confidence_threshold
                              else "medium" if final_score >= self.config.min_threshold
                              else "low"
        }
        
        # Add explanations
        reasoning["explanations"] = {
            "strengths": [
                metric for metric, score in metrics.dict().items()
                if score >= 0.8
            ],
            "weaknesses": [
                metric for metric, score in metrics.dict().items()
                if score < 0.6
            ]
        }
        
        return reasoning

    async def _semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Calculate semantic similarity between texts"""
        # Implementation would use embeddings model
        return 0.8  # Placeholder

    async def _extract_query_elements(
        self,
        query: str
    ) -> List[str]:
        """Extract key elements from query"""
        # Implementation would use NLP model
        return [query]  # Placeholder

    def _extract_key_info(self, text: str) -> set:
        """Extract key information from text"""
        # Implementation would use NLP model
        return set([text])  # Placeholder

    async def _extract_facts(
        self,
        text: str
    ) -> List[str]:
        """Extract facts from text"""
        # Implementation would use NLP model
        return [text]  # Placeholder

    async def _fact_similarity(
        self,
        fact1: str,
        fact2: str
    ) -> float:
        """Calculate similarity between facts"""
        # Implementation would use NLP model
        return 0.8  # Placeholder

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            "weights": self.config.weights,
            "min_threshold": self.config.min_threshold,
            "high_confidence_threshold": self.config.high_confidence_threshold,
            "use_dynamic_weights": self.config.use_dynamic_weights
        }
    
    @staticmethod
    def safe_mean(values: List[float]) -> float:
        """Calculate mean safely with empty lists"""
        if not values:
            return 0.0
        return float(np.mean(values))