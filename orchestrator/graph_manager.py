# risk_rag_system/orchestrator/graph_manager.py

from typing import Dict, List, Optional, Any
from loguru import logger
import networkx as nx
from pydantic import BaseModel

from nodes.base_node import BaseNode, NodeInput, NodeOutput

class GraphState(BaseModel):
    """Represents the current state of the graph execution"""
    current_node_id: str
    execution_path: List[str] = []
    accumulated_confidence: float = 1.0
    metadata: Dict[str, Any] = {}

class GraphManager:
    """Manages the execution flow between nodes"""
    
    def __init__(self):
        self.nodes: Dict[str, BaseNode] = {}
        self.graph = nx.DiGraph()
        self.start_node: Optional[str] = None
        logger.info("Initialized GraphManager")

    def add_node(self, node: BaseNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id)
        
        # If this is the first node, set it as start node
        if not self.start_node:
            self.start_node = node.node_id
            
        logger.info(f"Added node {node.node_id} to graph")

    def add_edge(self, from_node_id: str, to_node_id: str, condition: str) -> None:
        """Add an edge between nodes with a condition"""
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Both nodes must exist in the graph")
            
        self.graph.add_edge(from_node_id, to_node_id, condition=condition)
        self.nodes[from_node_id].add_next_node(condition, self.nodes[to_node_id])
        logger.debug(f"Added edge from {from_node_id} to {to_node_id} with condition: {condition}")

    async def execute(self, initial_input: NodeInput) -> NodeOutput:
        """Execute the graph starting from the start node"""
        if not self.start_node:
            raise ValueError("No start node defined")

        current_node_id = self.start_node
        current_input = initial_input
        state = GraphState(current_node_id=current_node_id)
        
        while True:
            try:
                current_node = self.nodes[current_node_id]
                logger.info(f"Executing node: {current_node_id}")
                
                # Process current node
                output = await current_node.process(current_input)
                state.execution_path.append(current_node_id)
                state.accumulated_confidence *= output.confidence_score

                # Route to next node
                next_node = await current_node.route(output)
                if not next_node:
                    logger.info(f"Workflow completed at node {current_node_id}")
                    return output
                
                # Update current node and input for next iteration
                current_node_id = next_node.node_id
                current_input = NodeInput(
                    content=output.content,
                    metadata={
                        **output.metadata,
                        "previous_node": current_node_id,
                        "accumulated_confidence": state.accumulated_confidence
                    }
                )
                
            except Exception as e:
                logger.error(f"Error in node {current_node_id}: {e}")
                raise

    def validate_graph(self) -> bool:
        """Validate graph structure"""
        if not self.start_node:
            logger.error("No start node defined")
            return False

        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                logger.error(f"Graph contains cycles: {cycles}")
                return False
        except nx.NetworkXNoCycle:
            pass

        # Check for unreachable nodes
        reachable = nx.descendants(self.graph, self.start_node)
        unreachable = set(self.nodes.keys()) - set([self.start_node]) - reachable
        if unreachable:
            logger.warning(f"Unreachable nodes found: {unreachable}")

        return True

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the graph"""
        return {
            "nodes": {node_id: node.get_state() for node_id, node in self.nodes.items()},
            "edges": list(self.graph.edges(data=True)),
            "start_node": self.start_node
        }