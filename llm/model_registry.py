# risk_rag_system/llm/model_registry.py

from typing import Dict, Any, Optional, List, Type
from pydantic import BaseModel
from loguru import logger
import torch
from pathlib import Path
import json
import hashlib
import time
from datetime import datetime

class ModelMetadata(BaseModel):
    """Metadata for registered models"""
    model_id: str
    name: str
    version: str
    type: str  # llm, embeddings, classifier, etc.
    source: str  # huggingface, local, api
    creation_date: str
    last_updated: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    hash: Optional[str] = None

class RegistryConfig(BaseModel):
    """Configuration for model registry"""
    registry_path: Path = Path("./models/registry")
    cache_path: Path = Path("./models/cache")
    max_cached_models: int = 5
    default_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sync_period: int = 3600  # seconds

class ModelRegistry:
    """Manages model registration and loading"""
    
    def __init__(self, config: Optional[RegistryConfig] = None):
        self.config = config or RegistryConfig()
        self.registered_models: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, Any] = {}
        self._initialize_registry()
        logger.info("Initialized ModelRegistry")

    def _initialize_registry(self) -> None:
        """Initialize registry storage"""
        self.config.registry_path.mkdir(parents=True, exist_ok=True)
        self.config.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing registrations
        registry_file = self.config.registry_path / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                for model_data in data.values():
                    self.registered_models[model_data["model_id"]] = ModelMetadata(
                        **model_data
                    )

    async def register_model(
        self,
        name: str,
        version: str,
        model_type: str,
        source: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> ModelMetadata:
        """Register a new model"""
        try:
            # Generate unique model ID
            model_id = hashlib.sha256(
                f"{name}_{version}_{model_type}_{source}".encode()
            ).hexdigest()[:12]
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                type=model_type,
                source=source,
                creation_date=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                config=config,
                metrics=metrics or {},
            )
            
            # Add to registry
            self.registered_models[model_id] = metadata
            
            # Save registry
            await self._save_registry()
            
            logger.info(f"Registered model {name} v{version} with ID {model_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    async def load_model(
        self,
        model_id: str,
        device: Optional[str] = None
    ) -> Any:
        """Load a registered model"""
        if model_id not in self.registered_models:
            raise ValueError(f"Model not found: {model_id}")
            
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
            
        try:
            metadata = self.registered_models[model_id]
            device = device or self.config.default_device
            
            # Load based on source
            if metadata.source == "huggingface":
                model = await self._load_from_huggingface(metadata, device)
            elif metadata.source == "local":
                model = await self._load_from_local(metadata, device)
            elif metadata.source == "api":
                model = await self._load_api_client(metadata)
            else:
                raise ValueError(f"Unsupported model source: {metadata.source}")
            
            # Cache model
            self.loaded_models[model_id] = model
            
            # Manage cache size
            if len(self.loaded_models) > self.config.max_cached_models:
                await self._cleanup_cache()
                
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise

    async def _load_from_huggingface(
        self,
        metadata: ModelMetadata,
        device: str
    ) -> Any:
        """Load model from HuggingFace"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Check cache first
        cache_path = self._get_cache_path(metadata)
        if cache_path.exists():
            model = AutoModelForCausalLM.from_pretrained(
                cache_path,
                device_map="auto" if device == "cuda" else None
            )
            tokenizer = AutoTokenizer.from_pretrained(cache_path)
        else:
            # Download and cache
            model = AutoModelForCausalLM.from_pretrained(
                metadata.name,
                device_map="auto" if device == "cuda" else None
            )
            tokenizer = AutoTokenizer.from_pretrained(metadata.name)
            
            # Save to cache
            model.save_pretrained(cache_path)
            tokenizer.save_pretrained(cache_path)
        
        return {"model": model, "tokenizer": tokenizer}

    async def _load_from_local(
        self,
        metadata: ModelMetadata,
        device: str
    ) -> Any:
        """Load model from local storage"""
        model_path = Path(metadata.config["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        # Load based on model type
        if metadata.type == "pytorch":
            return torch.load(model_path, map_location=device)
        elif metadata.type == "onnx":
            import onnxruntime as ort
            return ort.InferenceSession(str(model_path))
        else:
            raise ValueError(f"Unsupported local model type: {metadata.type}")

    async def _load_api_client(self, metadata: ModelMetadata) -> Any:
        """Load API client configuration"""
        return {
            "api_url": metadata.config["api_url"],
            "api_key": metadata.config.get("api_key"),
            "model_params": metadata.config.get("model_params", {})
        }

    async def update_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Update model metrics"""
        if model_id not in self.registered_models:
            raise ValueError(f"Model not found: {model_id}")
            
        metadata = self.registered_models[model_id]
        metadata.metrics.update(metrics)
        metadata.last_updated = datetime.now().isoformat()
        
        await self._save_registry()
        logger.info(f"Updated metrics for model {model_id}")

    async def _save_registry(self) -> None:
        """Save registry to disk"""
        registry_file = self.config.registry_path / "registry.json"
        
        registry_data = {
            model_id: metadata.dict()
            for model_id, metadata in self.registered_models.items()
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)

    def _get_cache_path(self, metadata: ModelMetadata) -> Path:
        """Get cache path for model"""
        return self.config.cache_path / metadata.model_id

    async def _cleanup_cache(self) -> None:
        """Clean up model cache"""
        # Get cache usage info
        cache_items = []
        for model_id in self.loaded_models:
            cache_path = self._get_cache_path(self.registered_models[model_id])
            if cache_path.exists():
                cache_items.append({
                    "model_id": model_id,
                    "size": sum(f.stat().st_size for f in cache_path.rglob("*")),
                    "last_accessed": cache_path.stat().st_atime
                })
        
        # Sort by last accessed time
        cache_items.sort(key=lambda x: x["last_accessed"])
        
        # Remove oldest items until we're under the limit
        while len(cache_items) > self.config.max_cached_models:
            item = cache_items.pop(0)
            model_id = item["model_id"]
            
            # Unload model
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                
            # Clear cache
            cache_path = self._get_cache_path(self.registered_models[model_id])
            if cache_path.exists():
                for f in cache_path.rglob("*"):
                    f.unlink()
                cache_path.rmdir()
                
            logger.info(f"Cleaned up cached model {model_id}")

    async def get_model_info(
        self,
        model_id: str
    ) -> Dict[str, Any]:
        """Get detailed model information"""
        if model_id not in self.registered_models:
            raise ValueError(f"Model not found: {model_id}")
            
        metadata = self.registered_models[model_id]
        cache_path = self._get_cache_path(metadata)
        
        return {
            **metadata.dict(),
            "is_loaded": model_id in self.loaded_models,
            "is_cached": cache_path.exists(),
            "cache_size": sum(f.stat().st_size for f in cache_path.rglob("*"))
            if cache_path.exists() else 0
        }

    async def list_models(
        self,
        model_type: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[ModelMetadata]:
        """List registered models with optional filtering"""
        models = list(self.registered_models.values())
        
        if model_type:
            models = [m for m in models if m.type == model_type]
        if source:
            models = [m for m in models if m.source == source]
            
        return sorted(models, key=lambda x: x.last_updated, reverse=True)

    def get_state(self) -> Dict[str, Any]:
        """Get current registry state"""
        return {
            "num_registered": len(self.registered_models),
            "num_loaded": len(self.loaded_models),
            "cache_path": str(self.config.cache_path),
            "registry_path": str(self.config.registry_path)
        }
    
    def cleanup(self) -> None:
        """Cleanup model resources"""
        try:
            # Unload models
            for model_id in list(self.loaded_models.keys()):
                if model_id in self.loaded_models:
                    del self.loaded_models[model_id]

            # Clear registry
            self.registered_models.clear()

            # Force GPU memory cleanup if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Cleaned up ModelRegistry resources")
        except Exception as e:
            logger.error(f"Error cleaning up ModelRegistry: {e}")