# risk_rag_system/indexing/raptor/recursive_summarizer.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration

@dataclass
class SummaryNode:
    """Node in the RAPTOR summary tree"""
    content: str
    level: int
    children: List['SummaryNode']
    metadata: Dict[str, Any]
    embedding: Optional[torch.Tensor] = None

class RaptorConfig(BaseModel):
    """Configuration for RAPTOR summarization"""
    max_tree_depth: int = 3
    summary_length_ratios: Dict[int, float] = Field(
        default_factory=lambda: {
            0: 0.3,  # Root level: 30% of original
            1: 0.5,  # Level 1: 50% of parent
            2: 0.7   # Level 2: 70% of parent
        }
    )
    min_chunk_size: int = 100
    max_chunk_size: int = 2048
    overlap_size: int = 50
    model_name: str = "google/pegasus-large"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class RaptorSummarizer:
    """Implements RAPTOR hierarchical summarization"""

    def __init__(self, config: Optional[RaptorConfig] = None):
        self.config = config or RaptorConfig()
        self._initialize_model()
        logger.info(f"Initialized RAPTOR summarizer with model: {self.config.model_name}")

    def _initialize_model(self) -> None:
        """Initialize the summarization model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqGeneration.from_pretrained(
                self.config.model_name
            )
            self.model.to(self.config.device)
            logger.info("Successfully loaded summarization model")
        except Exception as e:
            logger.error(f"Error initializing summarization model: {e}")
            raise

    async def build_summary_tree(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SummaryNode:
        """
        Build a hierarchical summary tree from text
        
        Args:
            text: Input text to summarize
            metadata: Optional metadata to attach to nodes
            
        Returns:
            Root node of summary tree
        """
        try:
            # Start recursive summarization from level 0
            root_node = await self._recursive_summarize(
                text,
                level=0,
                metadata=metadata or {}
            )
            logger.info("Successfully built summary tree")
            return root_node
        except Exception as e:
            logger.error(f"Error building summary tree: {e}")
            raise

    async def _recursive_summarize(
        self,
        text: str,
        level: int,
        metadata: Dict[str, Any]
    ) -> SummaryNode:
        """Recursively build summary tree"""
        if level >= self.config.max_tree_depth:
            return SummaryNode(text, level, [], metadata)

        # Generate summary for current level
        ratio = self.config.summary_length_ratios.get(level, 0.5)
        summary = await self._generate_summary(text, ratio)

        # Split text for next level if not at max depth
        if level + 1 < self.config.max_tree_depth:
            chunks = self._split_text(text)
            children = []
            for i, chunk in enumerate(chunks):
                child_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "parent_summary": summary
                }
                child = await self._recursive_summarize(
                    chunk,
                    level + 1,
                    child_metadata
                )
                children.append(child)
        else:
            children = []

        return SummaryNode(summary, level, children, metadata)

    async def _generate_summary(self, text: str, ratio: float) -> str:
        """Generate summary of specified length ratio"""
        try:
            # Calculate target length
            target_length = int(len(text.split()) * ratio)
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=self.config.max_chunk_size,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.config.device)

            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=target_length,
                min_length=min(target_length // 2, self.config.min_chunk_size),
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks for next level"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word)
            if current_length + word_length > self.config.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the summarizer"""
        return {
            "model_name": self.config.model_name,
            "max_tree_depth": self.config.max_tree_depth,
            "summary_ratios": self.config.summary_length_ratios,
            "device": self.config.device
        }