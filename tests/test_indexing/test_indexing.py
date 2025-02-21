# risk_rag_system/tests/test_indexing/test_indexing.py

import pytest
from typing import Dict, Any, List
import torch
import numpy as np
from pathlib import Path
import networkx as nx


from tests.test_base import (
    MockTestCase,
    EmbeddingTestMixin,
    requires_gpu,
    slow_test,
    with_timeout
)
from indexing.raptor.recursive_summarizer import RaptorSummarizer, RaptorConfig
from indexing.raptor.tree_builder import RaptorTreeBuilder, TreeBuilderConfig
from indexing.raptor.summary_indexer import RaptorIndexer, IndexerConfig
from indexing.hybrid_embeddings.bge_embedder import BGEEmbedder, EmbeddingConfig
from indexing.hybrid_embeddings.splade_embedder import SpladeProcessor, SpladeConfig
from indexing.raptor.recursive_summarizer import SummaryNode
class TestRaptorSummarizer(MockTestCase):
    """Tests for RAPTOR summarizer"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.config = RaptorConfig(
            max_tree_depth=3,
            summary_length_ratios={0: 0.3, 1: 0.5, 2: 0.7}
        )
        self.summarizer = RaptorSummarizer(self.config)
        
        # Test document
        self.test_doc = """
        This is a long test document that will be summarized recursively.
        It contains multiple paragraphs of information.
        Each paragraph discusses different topics.
        
        This is the second paragraph with more details.
        It provides additional context about the topics.
        The information here builds upon the first paragraph.
        
        Finally, this concluding paragraph wraps everything up.
        It summarizes the main points discussed above.
        This should be captured in the summary hierarchy.
        """
    
    @pytest.mark.asyncio
    async def test_build_summary_tree(self):
        """Test building summary tree"""
        metadata = {"source": "test"}
        
        root_node = await self.summarizer.build_summary_tree(
            self.test_doc,
            metadata
        )
        
        # Verify tree structure
        assert root_node.level == 0
        assert len(root_node.children) > 0
        assert root_node.metadata == metadata
        assert root_node.content != self.test_doc
        
        # Check levels
        levels = {0: [root_node]}
        queue = [(child, 1) for child in root_node.children]
        while queue:
            node, level = queue.pop(0)
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
            queue.extend((child, level + 1) for child in node.children)
        
        assert len(levels) <= self.config.max_tree_depth
        
        # Verify length ratios
        for level, nodes in levels.items():
            if level in self.config.summary_length_ratios:
                ratio = self.config.summary_length_ratios[level]
                for node in nodes:
                    expected_length = len(self.test_doc.split()) * ratio
                    actual_length = len(node.content.split())
                    assert actual_length <= expected_length * 1.1  # 10% tolerance

    @slow_test
    @pytest.mark.asyncio
    async def test_recursive_summarization(self):
        """Test recursive summarization process"""
        metadata = {"depth": 0}
        
        node = await self.summarizer._recursive_summarize(
            self.test_doc,
            level=0,
            metadata=metadata
        )
        
        # Check recursive structure
        def check_node(node, level):
            assert node.level == level
            assert isinstance(node.content, str)
            assert len(node.content) > 0
            assert isinstance(node.metadata, dict)
            if level < self.config.max_tree_depth - 1:
                assert len(node.children) > 0
                for child in node.children:
                    check_node(child, level + 1)
        
        check_node(node, 0)

    @pytest.mark.asyncio
    async def test_generate_summary(self):
        """Test summary generation"""
        text = "Test content for summary generation with specific ratio."
        ratio = 0.5
        
        summary = await self.summarizer._generate_summary(text, ratio)
        
        # Verify summary properties
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) < len(text)
        words_ratio = len(summary.split()) / len(text.split())
        assert abs(words_ratio - ratio) < 0.2  # 20% tolerance

class TestRaptorTreeBuilder(MockTestCase):
    """Tests for RAPTOR tree builder"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.config = TreeBuilderConfig(
            max_siblings=5,
            min_similarity_threshold=0.7
        )
        self.builder = RaptorTreeBuilder(self.config)
        
        # Test document
        self.test_doc = {
            "id": "test_doc",
            "content": """
            Test document for tree building.
            This contains multiple sections.
            Each section has unique content.
            
            Section two discusses different topics.
            It provides more information.
            The content here is distinct.
            
            Final section wraps up.
            It contains concluding remarks.
            """
        }
    
    @pytest.mark.asyncio
    async def test_build_tree(self):
        """Test building RAPTOR tree"""
        metadata = {"source": "test"}
        
        tree = await self.builder.build_tree(
            self.test_doc,
            metadata
        )
        
        # Verify tree properties
        assert tree.root is not None
        assert isinstance(tree.graph, nx.DiGraph)
        assert len(tree.level_summaries) > 0
        assert tree.metadata["document_id"] == self.test_doc["id"]
        
        # Check graph properties
        assert nx.is_directed_acyclic_graph(tree.graph)
        assert len(tree.graph.nodes) > 0
        assert len(tree.graph.edges) > 0

    @pytest.mark.asyncio
    async def test_add_cross_links(self):
        """Test cross-linking between nodes"""
        # Build basic tree first
        tree = await self.builder.build_tree(self.test_doc, {})
        
        # Get nodes at same level
        level_nodes = tree.level_summaries[1]  # Use level 1 for testing
        if len(level_nodes) >= 2:
            # Add embeddings manually for testing
            embeddings = torch.randn(len(level_nodes), 768)
            for node, emb in zip(level_nodes, embeddings):
                node.embedding = emb
            
            # Add cross-links
            await self.builder._add_cross_links(
                tree.graph,
                {1: level_nodes}
            )
            
            # Verify cross-links
            cross_links = [
                (u, v) for u, v, data in tree.graph.edges(data=True)
                if data.get("type") == "semantic"
            ]
            assert len(cross_links) > 0

    @requires_gpu
    @pytest.mark.asyncio
    async def test_add_embeddings(self):
        """Test adding embeddings to nodes"""
        tree = await self.builder.build_tree(self.test_doc, {})
        root_node = tree.root
        
        # Add embeddings
        await self.builder._add_embeddings(root_node)
        
        # Verify embeddings
        def check_embeddings(node):
            assert node.embedding is not None
            assert isinstance(node.embedding, torch.Tensor)
            assert node.embedding.shape[-1] == 768  # BGE embedding dimension
            for child in node.children:
                check_embeddings(child)
        
        check_embeddings(root_node)

class TestRaptorIndexer(MockTestCase, EmbeddingTestMixin):
    """Tests for RAPTOR indexer"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.config = IndexerConfig(
            collection_name="test_collection",
            distance_metric="cosine"
        )
        self.indexer = RaptorIndexer(self.config)
        
        # Test documents
        self.test_docs = [
            {
                "id": "doc1",
                "content": "Test document one with specific content."
            },
            {
                "id": "doc2",
                "content": "Another test document with different content."
            }
        ]
    
    @pytest.mark.asyncio
    async def test_index_document(self):
        """Test document indexing"""
        for doc in self.test_docs:
            tree = await self.indexer.index_document(doc)
            
            # Verify indexing results
            assert tree is not None
            assert isinstance(tree.root, SummaryNode)
            assert tree.metadata["document_id"] == doc["id"]

    @pytest.mark.asyncio
    async def test_query(self):
        """Test querying indexed documents"""
        # Index test documents first
        for doc in self.test_docs:
            await self.indexer.index_document(doc)
        
        # Perform query
        results = await self.indexer.query(
            query="test content",
            filters={"level": 0}
        )
        
        # Verify results
        assert len(results) > 0
        for result in results:
            assert result.content is not None
            assert result.level >= 0
            assert result.relevance_score > 0

    @pytest.mark.asyncio
    async def test_index_tree(self):
        """Test indexing RAPTOR tree"""
        # Build and index a tree
        doc = self.test_docs[0]
        tree = await self.indexer.tree_builder.build_tree(doc)
        await self.indexer._index_tree(tree)
        
        # Verify collection contents
        assert self.indexer.collection.count() > 0

class TestBGEEmbedder(MockTestCase, EmbeddingTestMixin):
    """Tests for BGE embedder"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.config = EmbeddingConfig(
            model_name="BAAI/bge-large-en-v1.5"
        )
        self.embedder = BGEEmbedder(self.config)
        
        self.test_texts = [
            "First test text for embedding generation.",
            "Second text with different content.",
            "Third example of text to embed."
        ]
    
    @requires_gpu
    @pytest.mark.asyncio
    async def test_encode(self):
        """Test text encoding"""
        embeddings = await self.embedder.encode(self.test_texts)
        
        # Verify embeddings
        self.assert_valid_embeddings(
            embeddings,
            expected_dim=1024,  # BGE large dimension
            batch_size=len(self.test_texts)
        )

    @requires_gpu
    @pytest.mark.asyncio
    async def test_encode_query(self):
        """Test query encoding"""
        query = "Test query for embedding"
        
        embedding = await self.embedder.encode_query(query)
        
        # Verify query embedding
        self.assert_valid_embeddings(
            embedding.unsqueeze(0),
            expected_dim=1024
        )

    @slow_test
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of texts"""
        # Create larger test set
        texts = self.test_texts * 10  # 30 texts total
        
        embeddings = await self.embedder.encode(texts, batch_size=8)
        
        # Verify batch processing results
        self.assert_valid_embeddings(
            embeddings,
            expected_dim=1024,
            batch_size=len(texts)
        )

class TestSpladeProcessor(MockTestCase):
    """Tests for SPLADE processor"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.config = SpladeConfig(
            top_k_terms=256,
            threshold=0.01
        )
        self.processor = SpladeProcessor(self.config)
        
        self.test_text = """
        Test document for SPLADE processing.
        This document contains multiple terms.
        Some terms should have higher weights.
        """
    
    @pytest.mark.asyncio
    async def test_process_text(self):
        """Test text processing"""
        term_weights, sparse_vector = await self.processor.process_text(
            self.test_text
        )
        
        # Verify term weights
        assert isinstance(term_weights, dict)
        assert len(term_weights) > 0
        assert all(isinstance(v, float) for v in term_weights.values())
        assert all(v >= self.config.threshold for v in term_weights.values())
        
        # Verify sparse vector
        assert isinstance(sparse_vector, torch.Tensor)
        assert sparse_vector.shape[0] == len(self.processor.vocab)
        assert torch.count_nonzero(sparse_vector) <= self.config.top_k_terms

    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test batch processing"""
        texts = [
            "First test document.",
            "Second document with different terms.",
            "Third document for processing."
        ]
        
        results = await self.processor.process_batch(texts, batch_size=2)
        
        # Verify batch results
        assert len(results) == len(texts)
        for weights in results:
            assert isinstance(weights, dict)
            assert len(weights) > 0
            assert all(v >= self.config.threshold for v in weights.values())

    @pytest.mark.asyncio
    async def test_compute_similarity(self):
        """Test similarity computation"""
        text1 = "Test document about specific topic."
        text2 = "Another document discussing same topic."
        
        weights1, _ = await self.processor.process_text(text1)
        weights2, _ = await self.processor.process_text(text2)
        
        similarity = await self.processor.compute_similarity(weights1, weights2)
        
        # Verify similarity score
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1