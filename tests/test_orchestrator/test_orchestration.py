# risk_rag_system/tests/test_orchestrator/test_orchestration.py

import pytest
from typing import Dict, Any, List
import asyncio
from pathlib import Path

from tests.test_base import (
    MockTestCase,
    IntegrationTestCase,
    with_timeout,
    slow_test
)
from orchestrator.graph_manager import GraphManager, GraphState
from orchestrator.confidence_router import ConfidenceRouter, RoutingDecision
from orchestrator.state_tracker import StateTracker
from nodes.base_node import BaseNode, NodeInput, NodeOutput
from nodes.retriever_node import RetrieverNode
from nodes.grader_node import GraderNode
from nodes.rewriter_node import RewriterNode
from nodes.generator_node import GeneratorNode

class TestGraphManager(MockTestCase):
    """Tests for GraphManager"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.graph_manager = GraphManager()
        
        # Create test nodes
        self.retriever = RetrieverNode("retriever")
        self.grader = GraderNode("grader")
        self.rewriter = RewriterNode("rewriter")
        self.generator = GeneratorNode("generator")
    
    @pytest.mark.asyncio
    async def test_add_nodes(self):
        """Test adding nodes to graph"""
        # Add nodes
        self.graph_manager.add_node(self.retriever)
        self.graph_manager.add_node(self.grader)
        self.graph_manager.add_node(self.rewriter)
        self.graph_manager.add_node(self.generator)
        
        # Verify nodes
        assert len(self.graph_manager.nodes) == 4
        assert "retriever" in self.graph_manager.nodes
        assert "grader" in self.graph_manager.nodes
        
        # Verify start node
        assert self.graph_manager.start_node == "retriever"
    
    @pytest.mark.asyncio
    async def test_add_edges(self):
        """Test adding edges between nodes"""
        # Add nodes
        self.graph_manager.add_node(self.retriever)
        self.graph_manager.add_node(self.grader)
        self.graph_manager.add_node(self.rewriter)
        
        # Add edges
        self.graph_manager.add_edge(
            "retriever",
            "grader",
            condition="default"
        )
        self.graph_manager.add_edge(
            "retriever",
            "rewriter",
            condition="low_confidence"
        )
        
        # Verify edges
        assert len(self.graph_manager.graph.edges) == 2
        assert self.graph_manager.graph.has_edge("retriever", "grader")
        assert self.graph_manager.graph.has_edge("retriever", "rewriter")
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """Test workflow execution"""
        # Setup workflow
        self.graph_manager.add_node(self.retriever)
        self.graph_manager.add_node(self.grader)
        self.graph_manager.add_node(self.generator)
        
        self.graph_manager.add_edge("retriever", "grader", "default")
        self.graph_manager.add_edge("grader", "generator", "default")
        
        # Execute workflow
        input_data = NodeInput(
            content="test query",
            metadata={"source": "test"}
        )
        
        output = await self.graph_manager.execute(input_data)
        
        # Verify execution
        assert isinstance(output, NodeOutput)
        assert len(output.metadata) > 0
    
    @pytest.mark.asyncio
    async def test_graph_validation(self):
        """Test graph validation"""
        # Add nodes without edges
        self.graph_manager.add_node(self.retriever)
        self.graph_manager.add_node(self.grader)
        
        # Validate graph
        is_valid = self.graph_manager.validate_graph()
        assert is_valid
        
        # Add cycle
        self.graph_manager.add_edge("retriever", "grader", "default")
        self.graph_manager.add_edge("grader", "retriever", "feedback")
        
        # Validate graph with cycle
        is_valid = self.graph_manager.validate_graph()
        assert not is_valid

class TestConfidenceRouter(MockTestCase):
    """Tests for ConfidenceRouter"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.router = ConfidenceRouter()
    
    @pytest.mark.asyncio
    async def test_routing_decision(self):
        """Test routing decision making"""
        current_node = "retriever"
        confidence_scores = {
            "retrieval": 0.8,
            "validation": 0.7
        }
        available_nodes = ["generator", "rewriter", "validator"]
        context = {"query_type": "factual"}
        
        decision = await self.router.make_routing_decision(
            current_node,
            confidence_scores,
            available_nodes,
            context
        )
        
        # Verify decision
        assert isinstance(decision, RoutingDecision)
        assert decision.next_node in available_nodes
        assert decision.confidence >= 0.0
        assert len(decision.fallbacks) > 0
    
    @pytest.mark.asyncio
    async def test_confidence_thresholds(self):
        """Test confidence threshold handling"""
        # Test high confidence
        high_scores = {
            "retrieval": 0.9,
            "validation": 0.85
        }
        decision = await self.router.make_routing_decision(
            "retriever",
            high_scores,
            ["generator", "validator"],
            {}
        )
        assert decision.next_node == "generator"
        
        # Test low confidence
        low_scores = {
            "retrieval": 0.4,
            "validation": 0.5
        }
        decision = await self.router.make_routing_decision(
            "retriever",
            low_scores,
            ["generator", "validator"],
            {}
        )
        assert decision.next_node == "validator"

class TestStateTracker(MockTestCase):
    """Tests for StateTracker"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.tracker = StateTracker()
    
    @pytest.mark.asyncio
    async def test_node_state_tracking(self):
        """Test node state tracking"""
        node_id = "test_node"
        
        # Update node state
        await self.tracker.update_node_state(
            node_id=node_id,
            status="active",
            metrics={"latency": 100}
        )
        
        # Verify state
        assert node_id in self.tracker.node_states
        state = self.tracker.node_states[node_id]
        assert state.status == "active"
        assert state.metrics["latency"] == 100
    
    @pytest.mark.asyncio
    async def test_session_tracking(self):
        """Test session tracking"""
        session_id = "test_session"
        
        # Start session
        await self.tracker.start_session(
            session_id=session_id,
            metadata={"query": "test"}
        )
        
        assert session_id in self.tracker.active_sessions
        
        # End session
        await self.tracker.end_session(
            session_id=session_id,
            status="success",
            metrics={"duration": 1.5}
        )
        
        assert session_id not in self.tracker.active_sessions
    
    @pytest.mark.asyncio
    async def test_system_metrics(self):
        """Test system metrics calculation"""
        # Add some test data
        for i in range(5):
            await self.tracker.update_node_state(
                f"node_{i}",
                status="active",
                metrics={"success": True}
            )
        
        # Calculate metrics
        await self.tracker._update_system_metrics()
        
        # Verify metrics
        assert self.tracker.system_metrics.total_processed > 0
        assert 0 <= self.tracker.system_metrics.success_rate <= 1
        assert self.tracker.system_metrics.throughput >= 0

class TestEndToEndWorkflow(IntegrationTestCase):
    """End-to-end workflow tests"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        
        # Initialize components
        self.graph_manager = GraphManager()
        self.router = ConfidenceRouter()
        self.tracker = StateTracker()
        
        # Set up workflow
        await self._setup_workflow()
    
    async def _setup_workflow(self):
        """Set up test workflow"""
        # Create nodes
        self.retriever = RetrieverNode("retriever")
        self.grader = GraderNode("grader")
        self.rewriter = RewriterNode("rewriter")
        self.generator = GeneratorNode("generator")
        
        # Add nodes to graph
        self.graph_manager.add_node(self.retriever)
        self.graph_manager.add_node(self.grader)
        self.graph_manager.add_node(self.rewriter)
        self.graph_manager.add_node(self.generator)
        
        # Add edges
        self.graph_manager.add_edge("retriever", "grader", "default")
        self.graph_manager.add_edge("grader", "generator", "default")
        self.graph_manager.add_edge("grader", "rewriter", "low_confidence")
        self.graph_manager.add_edge("rewriter", "retriever", "default")
    
    @slow_test
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow execution"""
        # Start test session
        session_id = "test_session"
        await self.tracker.start_session(
            session_id=session_id,
            metadata={"test_type": "end_to_end"}
        )
        
        try:
            # Create input
            input_data = NodeInput(
                content="What are the key risks in project X?",
                metadata={
                    "session_id": session_id,
                    "priority": "high"
                }
            )
            
            # Execute workflow
            output = await self.graph_manager.execute(input_data)
            
            # Verify output
            assert isinstance(output, NodeOutput)
            assert output.confidence_score >= self.router.thresholds.overall
            assert "session_id" in output.metadata
            
            # Verify state tracking
            system_health = await self.tracker.get_system_health()
            assert system_health["metrics"].success_rate > 0
            assert len(system_health["node_status"]) == 4
            
        finally:
            # End session
            await self.tracker.end_session(
                session_id=session_id,
                status="completed",
                metrics={
                    "duration": 1.0,
                    "nodes_visited": 3
                }
            )
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test workflow error handling"""
        # Inject error in node
        async def error_process(self, input_data):
            raise ValueError("Test error")
        
        self.retriever.process = error_process.__get__(self.retriever)
        
        # Execute workflow
        input_data = NodeInput(content="test query")
        
        with pytest.raises(ValueError):
            await self.graph_manager.execute(input_data)
        
        # Verify error tracking
        node_metrics = await self.tracker.get_node_metrics("retriever")
        assert node_metrics["error_count"] > 0
    
    @pytest.mark.asyncio
    async def test_parallel_executions(self):
        """Test parallel workflow executions"""
        num_executions = 3
        inputs = [
            NodeInput(
                content=f"query_{i}",
                metadata={"execution_id": i}
            )
            for i in range(num_executions)
        ]
        
        # Execute workflows in parallel
        tasks = [
            self.graph_manager.execute(input_data)
            for input_data in inputs
        ]
        
        outputs = await asyncio.gather(*tasks)
        
        # Verify outputs
        assert len(outputs) == num_executions
        assert all(isinstance(out, NodeOutput) for out in outputs)
        assert len(set(out.metadata["execution_id"] for out in outputs)) == num_executions
    
    @pytest.mark.asyncio
    async def test_confidence_based_routing(self):
        """Test confidence-based routing decisions"""
        # Test high confidence path
        high_conf_input = NodeInput(
            content="clear query",
            metadata={"expected_confidence": "high"}
        )
        
        output = await self.graph_manager.execute(high_conf_input)
        execution_path = output.metadata.get("execution_path", [])
        assert "generator" in execution_path
        assert "rewriter" not in execution_path
        
        # Test low confidence path
        low_conf_input = NodeInput(
            content="ambiguous query",
            metadata={"expected_confidence": "low"}
        )
        
        output = await self.graph_manager.execute(low_conf_input)
        execution_path = output.metadata.get("execution_path", [])
        assert "rewriter" in execution_path
        assert "retriever" in execution_path[execution_path.index("rewriter"):]
    
    @slow_test
    @pytest.mark.asyncio
    async def test_state_persistence(self):
        """Test state persistence across sessions"""
        # Execute first workflow
        input_1 = NodeInput(content="query_1")
        output_1 = await self.graph_manager.execute(input_1)
        
        # Save state
        state_1 = self.tracker.get_state()
        
        # Create new tracker instance
        new_tracker = StateTracker()
        
        # Execute second workflow
        input_2 = NodeInput(content="query_2")
        output_2 = await self.graph_manager.execute(input_2)
        
        # Get new state
        state_2 = new_tracker.get_state()
        
        # Verify state persistence
        assert state_2["total_processed"] > state_1["total_processed"]
        assert all(node in state_2["nodes"] for node in state_1["nodes"])
    
    @pytest.mark.asyncio
    async def test_workflow_optimization(self):
        """Test workflow optimization based on metrics"""
        # Execute several workflows to gather metrics
        for i in range(5):
            input_data = NodeInput(
                content=f"query_{i}",
                metadata={"complexity": i % 3}
            )
            await self.graph_manager.execute(input_data)
        
        # Get node metrics
        retriever_metrics = await self.tracker.get_node_metrics("retriever")
        grader_metrics = await self.tracker.get_node_metrics("grader")
        
        # Update router thresholds based on metrics
        new_thresholds = {
            "retrieval": retriever_metrics["avg_confidence"],
            "grading": grader_metrics["avg_confidence"]
        }
        self.router.update_thresholds(new_thresholds)
        
        # Verify optimized routing
        test_input = NodeInput(content="optimization test")
        output = await self.graph_manager.execute(test_input)
        
        assert output.confidence_score >= min(new_thresholds.values())
        assert "optimization_metrics" in output.metadata