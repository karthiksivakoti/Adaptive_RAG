# risk_rag_system/indexing/raptor/tree_builder.py

from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from loguru import logger
import networkx as nx
import torch
from pydantic import BaseModel

from .recursive_summarizer import SummaryNode, RaptorSummarizer
from ..hybrid_embeddings.bge_embedder import BGEEmbedder

@dataclass
class RaptorTree:
    """Complete RAPTOR tree structure"""
    root: SummaryNode
    graph: nx.DiGraph
    level_summaries: Dict[int, List[SummaryNode]]
    metadata: Dict[str, Any]

class TreeBuilderConfig(BaseModel):
    """Configuration for RAPTOR tree builder"""
    max_siblings: int = 5
    min_similarity_threshold: float = 0.7
    cross_link_levels: bool = True
    enable_semantic_clustering: bool = True

class RaptorTreeBuilder:
    """Builds and manages RAPTOR tree structures"""

    def __init__(
        self,
        config: Optional[TreeBuilderConfig] = None,
        summarizer: Optional[RaptorSummarizer] = None
    ):
        self.config = config or TreeBuilderConfig()
        self.summarizer = summarizer or RaptorSummarizer()
        self.embedder = BGEEmbedder()
        self.trees: List[RaptorTree] = []
        logger.info("Initialized RAPTOR Tree Builder")

    async def build_tree(
        self,
        document: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> RaptorTree:
        """
        Build a RAPTOR tree from a document
        
        Args:
            document: Document dictionary with content
            metadata: Optional metadata
            
        Returns:
            Complete RAPTOR tree structure
        """
        try:
            # Build summary tree
            root_node = await self.summarizer.build_summary_tree(
                document["content"],
                metadata or {}
            )

            # Create graph representation
            graph = self._create_graph(root_node)

            # Organize summaries by level
            level_summaries = self._organize_levels(root_node)

            # Add embeddings to nodes
            await self._add_embeddings(root_node)

            # Create cross-links if enabled
            if self.config.cross_link_levels:
                await self._add_cross_links(graph, level_summaries)

            tree = RaptorTree(
                root=root_node,
                graph=graph,
                level_summaries=level_summaries,
                metadata={
                    **(metadata or {}),
                    "document_id": document.get("id", ""),
                    "tree_depth": len(level_summaries)
                }
            )

            self.trees.append(tree)
            logger.info(f"Successfully built RAPTOR tree for document {document.get('id', '')}")
            return tree

        except Exception as e:
            logger.error(f"Error building RAPTOR tree: {e}")
            raise

    def _create_graph(self, root: SummaryNode) -> nx.DiGraph:
        """Create NetworkX graph from tree structure"""
        graph = nx.DiGraph()
        
        def add_nodes_edges(node: SummaryNode, parent_id: Optional[str] = None):
            node_id = id(node)
            graph.add_node(
                node_id,
                content=node.content,
                level=node.level,
                metadata=node.metadata
            )
            if parent_id is not None:
                graph.add_edge(parent_id, node_id)
            for child in node.children:
                add_nodes_edges(child, node_id)

        add_nodes_edges(root)
        return graph

    def _organize_levels(self, root: SummaryNode) -> Dict[int, List[SummaryNode]]:
        """Organize nodes by their levels"""
        levels: Dict[int, List[SummaryNode]] = {}
        
        def traverse(node: SummaryNode):
            if node.level not in levels:
                levels[node.level] = []
            levels[node.level].append(node)
            for child in node.children:
                traverse(child)

        traverse(root)
        return levels

    async def _add_embeddings(self, node: SummaryNode) -> None:
        """Add embeddings to all nodes"""
        # Generate embedding for current node
        node.embedding = await self.embedder.encode([node.content])
        
        # Recursively process children
        for child in node.children:
            await self._add_embeddings(child)

    async def _add_cross_links(
        self,
        graph: nx.DiGraph,
        level_summaries: Dict[int, List[SummaryNode]]
    ) -> None:
        """Add cross-links between similar nodes at same level"""
        for level, nodes in level_summaries.items():
            if len(nodes) <= 1:
                continue

            # Get embeddings for all nodes at this level
            embeddings = torch.stack([node.embedding for node in nodes])
            
            # Calculate similarity matrix
            similarities = torch.matmul(embeddings, embeddings.T)
            
            # Add edges for similar nodes
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    similarity = similarities[i, j].item()
                    if similarity >= self.config.min_similarity_threshold:
                        graph.add_edge(
                            id(nodes[i]),
                            id(nodes[j]),
                            weight=similarity,
                            type="semantic"
                        )

    def traverse_tree(
        self,
        tree: RaptorTree,
        order: str = "bfs"
    ) -> Generator[SummaryNode, None, None]:
        """Traverse tree in specified order"""
        if order == "bfs":
            traversal = nx.bfs_tree(tree.graph, id(tree.root))
        else:  # dfs
            traversal = nx.dfs_tree(tree.graph, id(tree.root))

        for node_id in traversal:
            node_data = tree.graph.nodes[node_id]
            yield SummaryNode(
                content=node_data["content"],
                level=node_data["level"],
                children=[],  # Children handled by traversal
                metadata=node_data["metadata"]
            )

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the tree builder"""
        return {
            "num_trees": len(self.trees),
            "config": self.config.dict(),
            "cross_linking_enabled": self.config.cross_link_levels
        }