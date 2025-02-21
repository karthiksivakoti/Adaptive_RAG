from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    split_by_semantic: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 2048

class ProcessorConfig(BaseModel):
    supported_formats: List[str] = [
        ".pdf", ".docx", ".doc", 
        ".xlsx", ".xls", 
        ".txt", ".jpg", ".png"
    ]
    extract_tables: bool = True
    ocr_enabled: bool = True
    preserve_formatting: bool = True
    max_file_size_mb: int = 50

class VectorStoreConfig(BaseModel):
    implementation: str = "chroma"
    collection_name: str = "documents"
    distance_metric: str = "cosine"
    persist_directory: str = "./data/vectorstore"

class ModelConfig(BaseModel):
    embedding_model: str = "BAAI/bge-large-en"
    main_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    routing_model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    quantization: str = "4bit"
    device: str = "cuda"  # or "cpu"
    max_tokens: int = 2048

class SystemConfig(BaseModel):
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./cache")
    temp_dir: Path = Path("./temp")
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.data_dir, self.cache_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

# Default configuration instance
config = SystemConfig()