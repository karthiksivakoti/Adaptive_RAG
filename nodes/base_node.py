# risk_rag_system/nodes/base_node.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel
from loguru import logger

class NodeInput(BaseModel):
    """Base class for node inputs"""
    content: Any
    metadata: Dict[str, Any] = {}

class NodeOutput(BaseModel):
    """Base class for node outputs"""
    content: Any
    confidence_score: float
    metadata: Dict[str, Any] = {}

class BaseNode(ABC):
    """Abstract base class for all nodes in the Risk RAG system"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.node_type = self.__class__.__name__.lower().replace('node', '')
        self.next_nodes: Dict[str, 'BaseNode'] = {}
        self._initialize_node()
        logger.info(f"Initialized {self.node_type} node with ID: {node_id}")

    @abstractmethod
    def _initialize_node(self) -> None:
        """Initialize any node-specific components"""
        pass

    @abstractmethod
    async def process(self, input_data: NodeInput) -> NodeOutput:
        """Process input data and return output"""
        pass

    def add_next_node(self, condition: str, node: 'BaseNode') -> None:
        """Add a next node with a routing condition"""
        self.next_nodes[condition] = node
        logger.debug(f"Added next node {node.node_id} with condition: {condition}")

    async def route(self, output: NodeOutput) -> Optional['BaseNode']:
        """Route to next node based on output"""
        if not self.next_nodes:
            return None

        # Default routing logic - can be overridden by specific nodes
        if output.confidence_score < 0.5:
            return self.next_nodes.get('low_confidence')
        return self.next_nodes.get('default')

    async def validate_input(self, input_data: NodeInput) -> bool:
        """Validate input data"""
        return True

    def get_state(self) -> Dict[str, Any]:
        """Get current node state"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "next_nodes": list(self.next_nodes.keys())
        }