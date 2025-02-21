# risk_rag_system/tests/test_llm/test_model_registry.py

import pytest
import torch
from pathlib import Path
import json
import time
from typing import Dict, Any
import asyncio

from tests.test_base import (
    MockTestCase,
    requires_gpu,
    slow_test,
    with_timeout,
    TempModelContext
)
from llm.model_registry import (
    ModelRegistry,
    RegistryConfig,
    ModelMetadata
)

class TestModelRegistry(MockTestCase):
    """Tests for ModelRegistry"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.config = RegistryConfig(
            registry_path=self.method_dir / "registry",
            cache_path=self.method_dir / "cache",
            max_cached_models=3
        )
        self.registry = ModelRegistry(self.config)
        
        # Test model configurations
        self.test_models = [
            {
                "name": "test-model-1",
                "version": "v1",
                "type": "llm",
                "source": "huggingface",
                "config": {
                    "model_id": "test/model1",
                    "revision": "main"
                }
            },
            {
                "name": "test-model-2",
                "version": "v1",
                "type": "embeddings",
                "source": "local",
                "config": {
                    "path": str(self.method_dir / "models" / "model2.pt")
                }
            }
        ]
    
    @pytest.mark.asyncio
    async def test_register_model(self):
        """Test model registration"""
        model_info = self.test_models[0]
        
        metadata = await self.registry.register_model(
            name=model_info["name"],
            version=model_info["version"],
            model_type=model_info["type"],
            source=model_info["source"],
            config=model_info["config"]
        )
        
        # Verify registration
        assert isinstance(metadata, ModelMetadata)
        assert metadata.name == model_info["name"]
        assert metadata.version == model_info["version"]
        assert metadata.model_id in self.registry.registered_models
        
        # Verify registry file
        registry_file = self.config.registry_path / "registry.json"
        assert registry_file.exists()
        
        with open(registry_file, 'r') as f:
            saved_data = json.load(f)
            assert metadata.model_id in saved_data
    
    @requires_gpu
    @pytest.mark.asyncio
    async def test_load_huggingface_model(self):
        """Test loading HuggingFace model"""
        # Register test model
        metadata = await self.registry.register_model(
            name="test-hf-model",
            version="v1",
            model_type="llm",
            source="huggingface",
            config={
                "model_id": "mistralai/Mistral-7B-v0.1",
                "revision": "main"
            }
        )
        
        # Load model
        model = await self.registry.load_model(metadata.model_id)
        
        # Verify loaded model
        assert "model" in model
        assert "tokenizer" in model
        assert model["model"].device.type == self.config.default_device
        
        # Verify cache
        cache_path = self.registry._get_cache_path(metadata)
        assert cache_path.exists()
    
    @pytest.mark.asyncio
    async def test_load_local_model(self):
        """Test loading local model"""
        # Create test model file
        model_dir = self.method_dir / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "test_model.pt"
        
        # Save dummy model
        test_model = torch.nn.Linear(10, 2)
        torch.save(test_model.state_dict(), model_path)
        
        # Register model
        metadata = await self.registry.register_model(
            name="test-local-model",
            version="v1",
            model_type="pytorch",
            source="local",
            config={"path": str(model_path)}
        )
        
        # Load model
        model = await self.registry.load_model(metadata.model_id)
        
        # Verify loaded model
        assert isinstance(model, dict)
        assert model.keys() == test_model.state_dict().keys()
    
    @pytest.mark.asyncio
    async def test_load_api_model(self):
        """Test loading API model configuration"""
        metadata = await self.registry.register_model(
            name="test-api-model",
            version="v1",
            model_type="llm",
            source="api",
            config={
                "api_url": "https://api.test.com/v1",
                "model_params": {"temperature": 0.7}
            }
        )
        
        # Load API config
        config = await self.registry.load_model(metadata.model_id)
        
        # Verify config
        assert "api_url" in config
        assert "model_params" in config
        assert config["api_url"] == "https://api.test.com/v1"
    
    @slow_test
    @pytest.mark.asyncio
    async def test_cache_management(self):
        """Test model cache management"""
        # Register multiple models
        models = []
        for i in range(5):  # More than max_cached_models
            metadata = await self.registry.register_model(
                name=f"test-model-{i}",
                version="v1",
                model_type="pytorch",
                source="local",
                config={"path": str(self.method_dir / f"model_{i}.pt")}
            )
            models.append(metadata)
        
        # Load all models
        for metadata in models:
            await self.registry.load_model(metadata.model_id)
        
        # Verify cache size
        assert len(self.registry.loaded_models) <= self.config.max_cached_models
        
        # Verify oldest models were unloaded
        assert models[-1].model_id in self.registry.loaded_models
        assert models[0].model_id not in self.registry.loaded_models
    
    @pytest.mark.asyncio
    async def test_update_metrics(self):
        """Test updating model metrics"""
        # Register test model
        metadata = await self.registry.register_model(
            name="test-model",
            version="v1",
            model_type="llm",
            source="local",
            config={},
            metrics={"accuracy": 0.8}
        )
        
        # Update metrics
        new_metrics = {
            "accuracy": 0.85,
            "latency": 100
        }
        await self.registry.update_metrics(metadata.model_id, new_metrics)
        
        # Verify updates
        updated_info = await self.registry.get_model_info(metadata.model_id)
        assert updated_info["metrics"]["accuracy"] == 0.85
        assert updated_info["metrics"]["latency"] == 100
        assert updated_info["last_updated"] > metadata.last_updated
    
    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing registered models"""
        # Register multiple models
        for model_info in self.test_models:
            await self.registry.register_model(
                name=model_info["name"],
                version=model_info["version"],
                model_type=model_info["type"],
                source=model_info["source"],
                config=model_info["config"]
            )
        
        # List all models
        all_models = await self.registry.list_models()
        assert len(all_models) == len(self.test_models)
        
        # List by type
        llm_models = await self.registry.list_models(model_type="llm")
        assert all(m.type == "llm" for m in llm_models)
        
        # List by source
        hf_models = await self.registry.list_models(source="huggingface")
        assert all(m.source == "huggingface" for m in hf_models)
    
    @pytest.mark.asyncio
    async def test_get_model_info(self):
        """Test getting model information"""
        metadata = await self.registry.register_model(
            name="test-model",
            version="v1",
            model_type="llm",
            source="local",
            config={}
        )
        
        # Get info before loading
        info = await self.registry.get_model_info(metadata.model_id)
        assert not info["is_loaded"]
        assert not info["is_cached"]
        
        # Load model
        await self.registry.load_model(metadata.model_id)
        
        # Get info after loading
        info = await self.registry.get_model_info(metadata.model_id)
        assert info["is_loaded"]
        assert info["is_cached"]
        assert info["cache_size"] > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_cache(self):
        """Test cache cleanup"""
        # Register and load models
        loaded_models = []
        for i in range(self.config.max_cached_models + 2):
            metadata = await self.registry.register_model(
                name=f"test-model-{i}",
                version="v1",
                model_type="llm",
                source="local",
                config={}
            )
            await self.registry.load_model(metadata.model_id)
            loaded_models.append(metadata.model_id)
            
            # Add small delay to ensure different access times
            await asyncio.sleep(0.1)
        
        # Verify cleanup
        await self.registry._cleanup_cache()
        
        # Check that oldest models were removed
        assert loaded_models[0] not in self.registry.loaded_models
        assert loaded_models[1] not in self.registry.loaded_models
        assert loaded_models[-1] in self.registry.loaded_models
    
    @pytest.mark.asyncio
    async def test_model_versioning(self):
        """Test model versioning"""
        # Register multiple versions
        model_name = "test-model"
        versions = ["v1", "v2", "v3"]
        
        for version in versions:
            await self.registry.register_model(
                name=model_name,
                version=version,
                model_type="llm",
                source="local",
                config={}
            )
        
        # List all versions
        models = await self.registry.list_models()
        model_versions = [m.version for m in models if m.name == model_name]
        
        assert sorted(model_versions) == sorted(versions)
        assert len(model_versions) == len(versions)
    
    @pytest.mark.asyncio
    async def test_registry_persistence(self):
        """Test registry persistence across instances"""
        # Register model
        original_metadata = await self.registry.register_model(
            name="test-model",
            version="v1",
            model_type="llm",
            source="local",
            config={}
        )
        
        # Create new registry instance
        new_registry = ModelRegistry(self.config)
        
        # Verify model is still registered
        assert original_metadata.model_id in new_registry.registered_models
        loaded_metadata = new_registry.registered_models[original_metadata.model_id]
        assert loaded_metadata.name == original_metadata.name
        assert loaded_metadata.version == original_metadata.version