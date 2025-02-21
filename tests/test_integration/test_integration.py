# risk_rag_system/tests/test_integration/test_integration.py

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import json
import shutil
import time
import torch

from tests.test_base import (
    IntegrationTestCase,
    slow_test,
    requires_gpu,
    with_timeout
)

from input_processing.document.processor import DocumentProcessor
from indexing.raptor.recursive_summarizer import RaptorSummarizer
from indexing.hybrid_embeddings.bge_embedder import BGEEmbedder
from indexing.hybrid_embeddings.splade_embedder import SpladeProcessor
from nodes.retriever_node import RetrieverNode
from nodes.grader_node import GraderNode
from nodes.generator_node import GeneratorNode
from llm.model_registry import ModelRegistry
from orchestrator.graph_manager import GraphManager
from orchestrator.state_tracker import StateTracker

class TestSystemIntegration(IntegrationTestCase):
    """System-wide integration tests"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        
        # Initialize all components
        self.doc_processor = DocumentProcessor()
        self.summarizer = RaptorSummarizer()
        self.embedder = BGEEmbedder()
        self.splade = SpladeProcessor()
        self.model_registry = ModelRegistry()
        self.graph_manager = GraphManager()
        self.state_tracker = StateTracker()
        
        # Create test data
        await self._setup_test_data()
        
        # Set up workflow
        await self._setup_workflow()
    
    async def _setup_test_data(self):
        """Set up test documents and data"""
        # Create test documents
        self.test_docs = [
            {
                "content": """
                Risk Assessment Report
                Project X has several key risks:
                1. Technical debt in legacy systems
                2. Resource constraints in Q3
                3. Compliance requirements
                
                Mitigation strategies are ...
                """,
                "metadata": {
                    "type": "report",
                    "priority": "high"
                }
            },
            {
                "content": """
                Monthly Update
                Progress on risk mitigation:
                - Technical debt: 30% reduced
                - Resources: New team members onboarded
                - Compliance: Initial audit completed
                
                Next steps include ...
                """,
                "metadata": {
                    "type": "update",
                    "priority": "medium"
                }
            }
        ]
        
        # Process and index documents
        self.processed_docs = []
        for doc in self.test_docs:
            # Save temp file
            doc_path = self.test_data_dir / f"doc_{len(self.processed_docs)}.txt"
            with open(doc_path, 'w') as f:
                f.write(doc["content"])
            
            # Process document
            processed = await self.doc_processor.process_file(
                doc_path,
                doc["metadata"]
            )
            self.processed_docs.append(processed)
    
    async def _setup_workflow(self):
        """Set up workflow nodes and connections"""
        # Create nodes
        self.retriever = RetrieverNode("retriever")
        self.grader = GraderNode("grader")
        self.generator = GeneratorNode("generator")
        
        # Add nodes to graph
        self.graph_manager.add_node(self.retriever)
        self.graph_manager.add_node(self.grader)
        self.graph_manager.add_node(self.generator)
        
        # Add edges
        self.graph_manager.add_edge("retriever", "grader", "default")
        self.graph_manager.add_edge("grader", "generator", "default")
    
    @slow_test
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete system workflow"""
        # Start tracking
        session_id = "integration_test"
        await self.state_tracker.start_session(
            session_id=session_id,
            metadata={"test_type": "integration"}
        )
        
        try:
            # Create query
            query = "What are the current risks in Project X and their mitigation status?"
            
            # Execute workflow
            input_data = {
                "query": query,
                "metadata": {
                    "session_id": session_id,
                    "request_type": "risk_analysis"
                }
            }
            
            output = await self.graph_manager.execute(input_data)
            
            # Verify output
            assert isinstance(output.content, str)
            assert "technical debt" in output.content.lower()
            assert "compliance" in output.content.lower()
            assert output.confidence_score >= 0.7
            
            # Verify state tracking
            state = await self.state_tracker.get_system_health()
            assert state["metrics"].success_rate > 0.8
            
        finally:
            # End tracking
            await self.state_tracker.end_session(
                session_id=session_id,
                status="completed",
                metrics={"duration": 1.0}
            )
    
    @requires_gpu
    @pytest.mark.asyncio
    async def test_model_interactions(self):
        """Test interactions between different models"""
        # Register and load models
        llm_metadata = await self.model_registry.register_model(
            name="test-mistral",
            version="v1",
            model_type="llm",
            source="huggingface",
            config={"model_id": "mistralai/Mistral-7B-v0.1"}
        )
        
        embedder_metadata = await self.model_registry.register_model(
            name="test-bge",
            version="v1",
            model_type="embeddings",
            source="huggingface",
            config={"model_id": "BAAI/bge-large-en-v1.5"}
        )
        
        # Load models
        llm = await self.model_registry.load_model(llm_metadata.model_id)
        embedder = await self.model_registry.load_model(embedder_metadata.model_id)
        
        # Test embeddings generation
        text = "Test content for embedding"
        embeddings = await self.embedder.encode([text])
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[1] == 1024  # BGE dimension
        
        # Test LLM generation
        response = await llm.generate("Summarize: " + text)
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_document_pipeline(self):
        """Test document processing and indexing pipeline"""
        # Process document
        doc_path = self.test_data_dir / "test_doc.txt"
        with open(doc_path, 'w') as f:
            f.write("Test document with risk information.\nSome risks include...")
        
        processed_doc = await self.doc_processor.process_file(
            doc_path,
            {"type": "test"}
        )
        
        # Generate summaries
        summary_tree = await self.summarizer.build_summary_tree(
            processed_doc.content,
            processed_doc.metadata
        )
        
        # Generate embeddings
        doc_embedding = await self.embedder.encode([processed_doc.content])
        summary_embedding = await self.embedder.encode([summary_tree.root.content])
        
        # Verify pipeline outputs
        assert isinstance(processed_doc.content, str)
        assert isinstance(summary_tree.root.content, str)
        assert isinstance(doc_embedding, torch.Tensor)
        assert isinstance(summary_embedding, torch.Tensor)
    
    @slow_test
    @pytest.mark.asyncio
    async def test_concurrent_workflows(self):
        """Test concurrent workflow executions"""
        num_workflows = 3
        
        # Create queries
        queries = [
            f"Query {i}: What are the risks and mitigation strategies?"
            for i in range(num_workflows)
        ]
        
        # Execute workflows concurrently
        tasks = []
        for i, query in enumerate(queries):
            input_data = {
                "query": query,
                "metadata": {
                    "session_id": f"concurrent_test_{i}",
                    "priority": "medium"
                }
            }
            tasks.append(self.graph_manager.execute(input_data))
        
        # Gather results
        outputs = await asyncio.gather(*tasks)
        
        # Verify outputs
        assert len(outputs) == num_workflows
        assert all(out.confidence_score >= 0.7 for out in outputs)
        assert len(set(out.metadata["session_id"] for out in outputs)) == num_workflows
        
        # Verify system metrics
        metrics = await self.state_tracker.get_system_metrics()
        assert metrics.throughput >= num_workflows / 60  # workflows per minute
        assert metrics.error_rate == 0.0
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test system error recovery capabilities"""
        # Inject error into retriever
        original_process = self.retriever.process
        error_count = 0
        
        async def faulty_process(self, input_data):
            nonlocal error_count
            if error_count < 2:  # Fail twice then recover
                error_count += 1
                raise ValueError("Simulated retrieval error")
            return await original_process(input_data)
        
        self.retriever.process = faulty_process.__get__(self.retriever)
        
        try:
            # Execute workflow
            input_data = {
                "query": "Test error recovery",
                "metadata": {"retry_enabled": True}
            }
            
            output = await self.graph_manager.execute(input_data)
            
            # Verify recovery
            assert output.confidence_score >= 0.7
            assert "error_recovery" in output.metadata
            assert output.metadata["error_recovery"]["retry_count"] == 2
            
        finally:
            # Restore original process
            self.retriever.process = original_process
    
    @pytest.mark.asyncio
    async def test_system_monitoring(self):
        """Test system monitoring and metrics collection"""
        # Execute several workflows
        for i in range(3):
            input_data = {
                "query": f"Test query {i}",
                "metadata": {"monitoring_test": True}
            }
            await self.graph_manager.execute(input_data)
        
        # Get system metrics
        system_health = await self.state_tracker.get_system_health()
        node_metrics = {}
        for node_id in self.graph_manager.nodes:
            node_metrics[node_id] = await self.state_tracker.get_node_metrics(node_id)
        
        # Verify monitoring
        assert system_health["metrics"].total_processed == 3
        assert all(len(metrics) > 0 for metrics in node_metrics.values())
        assert all("latency" in metrics for metrics in node_metrics.values())
        assert all("success_rate" in metrics for metrics in node_metrics.values())
    
    @pytest.mark.asyncio
    async def test_data_consistency(self):
        """Test data consistency across system components"""
        # Process and index test document
        test_content = """
        Critical Risk Alert
        Severity: High
        Impact Areas: Technical, Financial
        Details: System vulnerability detected...
        """
        
        doc_path = self.test_data_dir / "consistency_test.txt"
        with open(doc_path, 'w') as f:
            f.write(test_content)
        
        # Process document
        processed_doc = await self.doc_processor.process_file(
            doc_path,
            {"type": "alert", "priority": "high"}
        )
        
        # Generate summaries and embeddings
        summary_tree = await self.summarizer.build_summary_tree(
            processed_doc.content,
            processed_doc.metadata
        )
        
        doc_embedding = await self.embedder.encode([processed_doc.content])
        summary_embedding = await self.embedder.encode([summary_tree.root.content])
        
        # Query system
        query = "What are the critical risks and their impact areas?"
        
        input_data = {
            "query": query,
            "metadata": {"consistency_test": True}
        }
        
        output = await self.graph_manager.execute(input_data)
        
        # Verify consistency
        assert "technical" in output.content.lower()
        assert "financial" in output.content.lower()
        assert output.metadata["source_doc_id"] == processed_doc.file_hash
        assert "alert" in str(output.metadata["doc_type"]).lower()
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self):
        """Test system performance optimization"""
        # Track initial performance
        initial_metrics = await self.state_tracker.get_system_metrics()
        
        # Execute batch of queries
        queries = [
            "What are the main risks?",
            "Describe compliance requirements.",
            "List technical debt issues."
        ]
        
        start_time = time.time()
        
        for query in queries:
            await self.graph_manager.execute({
                "query": query,
                "metadata": {"performance_test": True}
            })
        
        execution_time = time.time() - start_time
        
        # Get optimized metrics
        optimized_metrics = await self.state_tracker.get_system_metrics()
        
        # Verify optimization
        assert optimized_metrics.throughput >= initial_metrics.throughput
        assert execution_time / len(queries) <= 5.0  # Max 5 seconds per query
        
        # Check cache efficiency
        cache_metrics = await self.state_tracker.get_node_metrics("retriever", time_range=300)
        assert cache_metrics["cache_hit_rate"] > 0.5  # At least 50% cache hits
    
    def teardown_method(self):
        """Clean up after tests"""
        # Clean up test data
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
        
        # Clean up models
        if hasattr(self, 'model_registry'):
            self.model_registry.cleanup()
        
        # Clean up other resources
        super().teardown_method()