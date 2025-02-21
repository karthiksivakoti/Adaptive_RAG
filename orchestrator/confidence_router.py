# risk_rag_system/orchestrator/confidence_router.py

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel
from loguru import logger
import numpy as np

class RoutingDecision(BaseModel):
    """Structure for routing decisions"""
    next_node: str
    confidence: float
    reasoning: str
    fallbacks: List[str]

class ConfidenceThresholds(BaseModel):
    """Confidence thresholds for different operations"""
    retrieval: float = 0.7
    generation: float = 0.8
    validation: float = 0.75
    grading: float = 0.6
    overall: float = 0.7

class ConfidenceRouter:
    """Routes workflow based on confidence scores"""
    
    def __init__(self):
        self.thresholds = ConfidenceThresholds()
        self.confidence_history: Dict[str, List[float]] = {}
        self.routing_cache: Dict[str, RoutingDecision] = {}
        logger.info("Initialized ConfidenceRouter")

    async def make_routing_decision(
        self,
        current_node: str,
        confidence_scores: Dict[str, float],
        available_nodes: List[str],
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """Make routing decision based on confidence scores"""
        try:
            # Update confidence history
            self._update_history(current_node, confidence_scores)
            
            # Check cache for recent similar decisions
            cache_key = self._generate_cache_key(
                current_node,
                confidence_scores,
                context
            )
            
            if cache_key in self.routing_cache:
                logger.debug(f"Using cached routing decision for {cache_key}")
                return self.routing_cache[cache_key]
            
            # Calculate aggregate confidence
            aggregate_confidence = self._calculate_aggregate_confidence(
                confidence_scores
            )
            
            # Get primary and fallback nodes
            next_node, fallbacks = self._determine_next_nodes(
                current_node,
                aggregate_confidence,
                available_nodes,
                context
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                current_node,
                next_node,
                confidence_scores,
                context
            )
            
            decision = RoutingDecision(
                next_node=next_node,
                confidence=aggregate_confidence,
                reasoning=reasoning,
                fallbacks=fallbacks
            )
            
            # Cache decision
            self.routing_cache[cache_key] = decision
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making routing decision: {e}")
            raise

    def _update_history(
        self,
        node: str,
        scores: Dict[str, float]
    ) -> None:
        """Update confidence score history"""
        if node not in self.confidence_history:
            self.confidence_history[node] = []
            
        # Calculate average confidence for this update
        avg_confidence = sum(scores.values()) / len(scores)
        self.confidence_history[node].append(avg_confidence)
        
        # Keep only recent history
        if len(self.confidence_history[node]) > 100:
            self.confidence_history[node] = self.confidence_history[node][-100:]

    def _generate_cache_key(
        self,
        node: str,
        scores: Dict[str, float],
        context: Dict[str, Any]
    ) -> str:
        """Generate cache key for routing decision"""
        # Create key from relevant components
        score_str = "_".join(f"{k}:{v:.2f}" for k, v in sorted(scores.items()))
        context_keys = "_".join(sorted(context.keys()))
        return f"{node}_{score_str}_{context_keys}"

    def _calculate_aggregate_confidence(
        self,
        scores: Dict[str, float]
    ) -> float:
        """Calculate aggregate confidence score"""
        # Apply weights based on operation type
        weights = {
            "retrieval": 0.3,
            "generation": 0.25,
            "validation": 0.25,
            "grading": 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for op_type, score in scores.items():
            if op_type in weights:
                weight = weights[op_type]
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return sum(scores.values()) / len(scores)
            
        return weighted_sum / total_weight

    def _determine_next_nodes(
        self,
        current_node: str,
        confidence: float,
        available_nodes: List[str],
        context: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Determine next node and fallbacks"""
        # Define node transitions
        transitions = {
            "retriever": {
                "high": "generator",
                "low": "rewriter",
                "fallback": "web_search"
            },
            "generator": {
                "high": "grader",
                "low": "validator",
                "fallback": "rewriter"
            },
            "grader": {
                "high": "output",
                "low": "validator",
                "fallback": "retriever"
            }
        }
        
        # Get transition options
        node_options = transitions.get(current_node, {})
        
        # Determine primary next node
        if confidence >= self.thresholds.overall:
            next_node = node_options.get("high", "output")
        else:
            next_node = node_options.get("low", "validator")
        
        # Determine fallbacks
        fallbacks = [
            node for node in [
                node_options.get("fallback"),
                "validator",
                "retriever"
            ]
            if node in available_nodes and node != next_node
        ]
        
        return next_node, fallbacks

    def _generate_reasoning(
        self,
        current_node: str,
        next_node: str,
        scores: Dict[str, float],
        context: Dict[str, Any]
    ) -> str:
        """Generate reasoning for routing decision"""
        # Analyze confidence pattern
        trend = self._analyze_confidence_trend(current_node)
        
        # Build reasoning string
        reasoning_parts = [
            f"Current node ({current_node}) scores: " +
            ", ".join(f"{k}={v:.2f}" for k, v in scores.items()),
            f"Confidence trend: {trend}",
            f"Selected next node: {next_node}"
        ]
        
        if context:
            reasoning_parts.append(
                "Context factors: " +
                ", ".join(f"{k}={v}" for k, v in context.items())
            )
            
        return " | ".join(reasoning_parts)

    def _analyze_confidence_trend(self, node: str) -> str:
        """Analyze trend in confidence scores"""
        history = self.confidence_history.get(node, [])
        
        if len(history) < 2:
            return "insufficient_data"
            
        # Calculate trend
        recent = np.mean(history[-5:]) if len(history) >= 5 else history[-1]
        overall = np.mean(history)
        
        if abs(recent - overall) < 0.05:
            return "stable"
        return "improving" if recent > overall else "declining"

    def update_thresholds(
        self,
        new_thresholds: Dict[str, float]
    ) -> None:
        """Update confidence thresholds"""
        current = self.thresholds.dict()
        
        for key, value in new_thresholds.items():
            if key in current:
                setattr(self.thresholds, key, value)
                
        logger.info("Updated confidence thresholds")

    def clear_cache(self) -> None:
        """Clear routing cache"""
        self.routing_cache.clear()
        logger.info("Cleared routing cache")

    def get_state(self) -> Dict[str, Any]:
        """Get current state of router"""
        return {
            "thresholds": self.thresholds.dict(),
            "cache_size": len(self.routing_cache),
            "history_nodes": list(self.confidence_history.keys())
        }