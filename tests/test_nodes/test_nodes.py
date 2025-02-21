# risk_rag_system/tests/test_nodes/test_nodes.py

import pytest
from typing import Dict, Any
import torch
from pathlib import Path

from tests.test_base import (
    MockTestCase,
    NodeTestMixin,
    requires_gpu,
    slow_test
)
from nodes.retriever_node import RetrieverNode, RetrieverNodeInput
from nodes.grader_node import GraderNode, GraderNodeInput
from nodes.rewriter_node import RewriterNode, RewriterNodeInput
from nodes.generator_node import GeneratorNode, GeneratorNodeInput
from nodes.web_search_node import WebSearchNode, WebSearchInput

class TestRetrieverNode(MockTestCase, NodeTestMixin):
    """Tests for RetrieverNode"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.node = await self.setup_test_node(RetrieverNode)
        
        # Mock embeddings
        self.mock_embeddings = torch.randn(5, 768)
        self.mock_embedder.encode.return_value = self.mock_embeddings
    
    @pytest.mark.asyncio
    async def test_process_basic_query(self):
        """Test basic query processing"""
        input_data = RetrieverNodeInput(
            query="test query",
            filters={},
            top_k=3
        )
        
        output = await self.node.process(input_data)
        self.assert_valid_node_output(output)
        assert len(output.retrieved_documents) == 3
        
    @pytest.mark.asyncio
    async def test_process_with_filters(self):
        """Test query processing with filters"""
        input_data = RetrieverNodeInput(
            query="test query",
            filters={"type": "document"},
            top_k=5
        )
        
        output = await self.node.process(input_data)
        self.assert_valid_node_output(output)
        
    @requires_gpu
    @pytest.mark.asyncio
    async def test_dense_retrieval(self):
        """Test dense retrieval component"""
        query = "test query"
        query_embedding = torch.randn(768)
        self.mock_embedder.encode.return_value = query_embedding
        
        results = await self.node._dense_retrieval(
            RetrieverNodeInput(query=query)
        )
        assert len(results) > 0
        
    @pytest.mark.asyncio
    async def test_sparse_retrieval(self):
        """Test sparse retrieval component"""
        query = "test query"
        
        results = await self.node._sparse_retrieval(
            RetrieverNodeInput(query=query)
        )
        assert len(results) > 0
        
    @pytest.mark.asyncio
    async def test_hybrid_fusion(self):
        """Test hybrid fusion of results"""
        dense_results = [
            {"content": "doc1", "score": 0.9},
            {"content": "doc2", "score": 0.8}
        ]
        sparse_results = [
            {"content": "doc2", "score": 0.85},
            {"content": "doc3", "score": 0.7}
        ]
        
        fused_results = self.node._hybrid_fusion(dense_results, sparse_results)
        assert len(fused_results) >= 2
        assert fused_results[0].score >= fused_results[1].score

class TestGraderNode(MockTestCase, NodeTestMixin):
    """Tests for GraderNode"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.node = await self.setup_test_node(GraderNode)
    
    @pytest.mark.asyncio
    async def test_grade_document(self):
        """Test document grading"""
        query = "test query"
        document = {
            "content": "test content",
            "metadata": {"source": "test"}
        }
        
        relevance, hallucination = await self.node._grade_document(
            query=query,
            document=document,
            context="test context"
        )
        
        assert 0 <= relevance <= 1
        assert 0 <= hallucination <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_context_coverage(self):
        """Test context coverage calculation"""
        query = "test query"
        documents = [
            {"content": "relevant test content"},
            {"content": "unrelated content"}
        ]
        
        coverage = self.node._calculate_context_coverage(query, documents)
        assert 0 <= coverage <= 1

class TestRewriterNode(MockTestCase, NodeTestMixin):
    """Tests for RewriterNode"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.node = await self.setup_test_node(RewriterNode)
    
    @pytest.mark.asyncio
    async def test_rewrite_query(self):
        """Test query rewriting"""
        input_data = RewriterNodeInput(
            original_query="test query",
            grading_metrics={"relevance": 0.5},
            rewrite_reason="low relevance",
            failed_attempts=[]
        )
        
        output = await self.node.process(input_data)
        self.assert_valid_node_output(output)
        assert output.rewritten_query != input_data.original_query
    
    @pytest.mark.asyncio
    async def test_query_decomposition(self):
        """Test query decomposition"""
        query = "complex test query with multiple aspects"
        
        decomposition = await self.node._decompose_query(
            query=query,
            context="test context"
        )
        assert len(decomposition.sub_queries) > 1

class TestGeneratorNode(MockTestCase, NodeTestMixin):
    """Tests for GeneratorNode"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.node = await self.setup_test_node(GeneratorNode)
    
    @pytest.mark.asyncio
    async def test_generate_content(self):
        """Test content generation"""
        input_data = GeneratorNodeInput(
            query="test query",
            validated_content=[{"content": "test content"}],
            target_length=100
        )
        
        output = await self.node.process(input_data)
        self.assert_valid_node_output(output)
        assert len(output.generated_content) > 0
    
    @pytest.mark.asyncio
    async def test_content_plan_creation(self):
        """Test content plan creation"""
        query = "test query"
        validated_content = [{"content": "test content"}]
        
        plan = await self.node._create_content_plan(
            query=query,
            validated_content=validated_content,
            style="academic",
            constraints={}
        )
        assert len(plan.sections) > 0

class TestWebSearchNode(MockTestCase, NodeTestMixin):
    """Tests for WebSearchNode"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.node = await self.setup_test_node(WebSearchNode)
    
    @pytest.mark.asyncio
    async def test_web_search(self):
        """Test web search functionality"""
        input_data = WebSearchInput(
            query="test query",
            num_results=3
        )
        
        output = await self.node.process(input_data)
        self.assert_valid_node_output(output)
        assert len(output.results) > 0
    
    @pytest.mark.asyncio
    async def test_result_scoring(self):
        """Test search result scoring"""
        results = [
            {
                "title": "Test Result",
                "snippet": "Test content",
                "url": "http://test.com"
            }
        ]
        
        scored_results = await self.node._score_results(
            results=results,
            query="test query",
            min_relevance=0.5
        )
        assert len(scored_results) > 0
        assert all(0 <= r.relevance_score <= 1 for r in scored_results)