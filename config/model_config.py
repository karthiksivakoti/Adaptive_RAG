# risk_rag_system/config/model_config.py

from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
from pydantic import BaseModel, Field, validator
from enum import Enum
from loguru import logger

class ModelType(str, Enum):
    """Types of models in the system"""
    LLM = "llm"
    EMBEDDINGS = "embeddings"
    CLASSIFIER = "classifier"
    TOKENIZER = "tokenizer"
    SUMMARIZER = "summarizer"

class ModelSource(str, Enum):
    """Sources for model loading"""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    API = "api"

class QuantizationType(str, Enum):
    """Quantization methods"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ = "gptq"
    GGML = "ggml"

class ModelDeviceConfig(BaseModel):
    """Configuration for model device placement"""
    device_map: str = "auto"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16" if torch.cuda.is_available() else "float32"
    enable_flash_attention: bool = True
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = True

class CacheConfig(BaseModel):
    """Configuration for model caching"""
    cache_dir: Path = Field(default=Path("./cache/models"))
    max_memory_gb: float = 0.9  # Percentage of available GPU memory
    offload_folder: Optional[Path] = None
    preload_modules: List[str] = []
    enable_checkpoint: bool = True

class APIConfig(BaseModel):
    """Configuration for API-based models"""
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    org_id: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10

class ModelConfig(BaseModel):
    """Base configuration for all models"""
    name: str
    type: ModelType
    source: ModelSource
    revision: Optional[str] = None
    token: Optional[str] = None
    quantization: QuantizationType = QuantizationType.NONE
    device_config: ModelDeviceConfig = Field(default_factory=ModelDeviceConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    api_config: Optional[APIConfig] = None

class LLMConfig(ModelConfig):
    """Configuration for LLM models"""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    do_sample: bool = True
    use_cache: bool = True
    max_batch_size: int = 8
    truncation_strategy: str = "longest_first"

class EmbeddingConfig(ModelConfig):
    """Configuration for embedding models"""
    max_length: int = 512
    normalize_embeddings: bool = True
    pooling_strategy: str = "mean"
    embedding_dim: int = 768
    instruction_template: Optional[str] = None

class TokenizerConfig(ModelConfig):
    """Configuration for tokenizer models"""
    padding_side: str = "right"
    truncation_side: str = "right"
    model_max_length: int = 2048
    special_tokens: Dict[str, str] = {}
    add_bos_token: bool = True
    add_eos_token: bool = True

class SystemModelConfig(BaseModel):
    """Complete model configuration for the system"""
    # Main LLM models
    primary_llm: LLMConfig = Field(
        default=LLMConfig(
            name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            type=ModelType.LLM,
            source=ModelSource.HUGGINGFACE,
            quantization=QuantizationType.GPTQ,
            max_tokens=4096,
            device_config=ModelDeviceConfig(
                device_map="auto",
                enable_flash_attention=True
            )
        )
    )
    
    fallback_llm: LLMConfig = Field(
        default=LLMConfig(
            name="mistralai/Mistral-7B-Instruct-v0.1",
            type=ModelType.LLM,
            source=ModelSource.HUGGINGFACE,
            quantization=QuantizationType.INT4,
            max_tokens=2048
        )
    )
    
    # Embedding models
    primary_embeddings: EmbeddingConfig = Field(
        default=EmbeddingConfig(
            name="BAAI/bge-large-en-v1.5",
            type=ModelType.EMBEDDINGS,
            source=ModelSource.HUGGINGFACE,
            embedding_dim=1024,
            normalize_embeddings=True,
            instruction_template="Represent this sentence for retrieval: {}"
        )
    )
    
    sparse_embeddings: EmbeddingConfig = Field(
        default=EmbeddingConfig(
            name="naver/splade-cocondenser-ensembled",
            type=ModelType.EMBEDDINGS,
            source=ModelSource.HUGGINGFACE,
            normalize_embeddings=False
        )
    )
    
    # Tokenizer configuration
    system_tokenizer: TokenizerConfig = Field(
        default=TokenizerConfig(
            name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            type=ModelType.TOKENIZER,
            source=ModelSource.HUGGINGFACE,
            model_max_length=4096
        )
    )
    
    # Special tokens and templates
    special_tokens: Dict[str, str] = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "sep_token": "<sep>",
        "user_token": "<|user|>",
        "assistant_token": "<|assistant|>",
        "system_token": "<|system|>"
    }
    
    chat_template: str = """{system_token}You are a helpful AI assistant.{sep_token}
{user_token}{query}{sep_token}
{assistant_token}"""
    
    # System settings
    max_model_memory: float = 0.9  # Maximum GPU memory usage (90%)
    enable_model_offloading: bool = True
    model_swap_strategy: str = "adaptive"  # adaptive, fixed, or disabled
    model_load_timeout: int = 30  # seconds
    
    # Load balancing settings
    max_concurrent_requests: int = 10
    request_timeout: int = 60
    batch_scheduling: bool = True
    priority_users: List[str] = []
    
    # Caching and optimization
    global_cache_dir: Path = Path("./cache")
    shared_cache_size_gb: float = 4.0
    optimizer_settings: Dict[str, Any] = {
        "use_bettertransformer": True,
        "enable_memory_efficient_attention": True,
        "enable_sequential_cpu_offload": False,
        "enable_attention_slicing": "auto"
    }
    
    @validator("global_cache_dir", pre=True)
    def create_cache_dir(cls, v):
        """Ensure cache directory exists"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_model_config(
        self,
        model_type: str,
        variant: str = "primary"
    ) -> ModelConfig:
        """Get configuration for specific model type"""
        configs = {
            "llm": {
                "primary": self.primary_llm,
                "fallback": self.fallback_llm
            },
            "embeddings": {
                "primary": self.primary_embeddings,
                "sparse": self.sparse_embeddings
            },
            "tokenizer": {
                "primary": self.system_tokenizer
            }
        }
        
        if model_type not in configs:
            raise ValueError(f"Unknown model type: {model_type}")
            
        if variant not in configs[model_type]:
            raise ValueError(f"Unknown variant {variant} for type {model_type}")
            
        return configs[model_type][variant]
    
    def format_template(
        self,
        template_type: str,
        **kwargs
    ) -> str:
        """Format system template with parameters"""
        templates = {
            "chat": self.chat_template
        }
        
        if template_type not in templates:
            raise ValueError(f"Unknown template type: {template_type}")
            
        # Replace special tokens
        kwargs.update(self.special_tokens)
        return templates[template_type].format(**kwargs)
    
    def get_device_config(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """Get device configuration for model"""
        configs = {
            self.primary_llm.name: {
                "device_map": "auto",
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True
            },
            self.primary_embeddings.name: {
                "device_map": "auto",
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
            }
        }
        
        return configs.get(model_name, {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch_dtype": torch.float32
        })
    
    def get_state(self) -> Dict[str, Any]:
        """Get current configuration state"""
        return {
            "models": {
                "primary_llm": self.primary_llm.dict(),
                "embeddings": self.primary_embeddings.dict(),
                "tokenizer": self.system_tokenizer.dict()
            },
            "system": {
                "cache_dir": str(self.global_cache_dir),
                "max_memory": self.max_model_memory,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }