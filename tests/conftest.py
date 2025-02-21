# risk_rag_system/tests/conftest.py

import pytest
from pathlib import Path
import shutil
import torch
import numpy as np
import random
import os
from typing import Generator
import logging

# Set random seeds for reproducibility
def pytest_configure():
    """Configure test environment"""
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set environment variables
    os.environ["TEST_MODE"] = "true"
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)

@pytest.fixture(scope="session")
def test_dir() -> Generator[Path, None, None]:
    """Provide test directory"""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    yield test_dir
    
    # Clean up
    if test_dir.exists():
        shutil.rmtree(test_dir)

@pytest.fixture(scope="session")
def model_cache_dir() -> Generator[Path, None, None]:
    """Provide model cache directory"""
    cache_dir = Path(__file__).parent / "model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    yield cache_dir
    
    # Clean up
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

@pytest.fixture(scope="session")
def vector_store_dir() -> Generator[Path, None, None]:
    """Provide vector store directory"""
    store_dir = Path(__file__).parent / "vector_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    
    yield store_dir
    
    # Clean up
    if store_dir.exists():
        shutil.rmtree(store_dir)

@pytest.fixture(scope="session")
def test_device() -> str:
    """Provide test device"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache before and after each test"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def pytest_collection_modifyitems(items):
    """Modify test collection"""
    # Mark slow tests
    for item in items:
        if "integration" in item.nodeid or "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Skip GPU tests if no GPU available
        if "gpu" in item.nodeid and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="No GPU available"))

# Add custom markers
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark as integration test")