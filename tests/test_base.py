# risk_rag_system/tests/test_base.py

import pytest
import asyncio
from typing import Dict, Any, Optional, Generator, List
from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, patch
import json
import numpy as np
import torch
from loguru import logger
from datetime import datetime, timedelta
from typing import AsyncGenerator

# Test configuration
class TestConfig:
    """Configuration for test environment"""
    TEST_DATA_DIR = Path("./tests/data")
    TEMP_DIR = Path("./tests/temp")
    MODEL_CACHE_DIR = Path("./tests/model_cache")
    VECTOR_STORE_DIR = Path("./tests/vector_store")
    TEST_TIMEOUT = 30  # seconds

# Base fixtures
@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Provide test configuration"""
    return TestConfig()

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def temp_dir(test_config: TestConfig) -> AsyncGenerator[Path, None]:
    """Provide temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp(dir=test_config.TEMP_DIR))
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture(scope="session")
def test_documents() -> List[Dict[str, Any]]:
    """Provide test document data"""
    return [
        {
            "content": "This is a test document for risk analysis.",
            "metadata": {
                "source": "test",
                "type": "text",
                "risk_level": "medium"
            }
        },
        {
            "content": "Another test document with different content.",
            "metadata": {
                "source": "test",
                "type": "text",
                "risk_level": "low"
            }
        }
    ]

@pytest.fixture(scope="session")
def test_embeddings() -> torch.Tensor:
    """Provide test embeddings"""
    return torch.randn(10, 768)  # Example dimensions

@pytest.fixture(scope="session")
def mock_llm() -> MagicMock:
    """Provide mock LLM for testing"""
    mock = MagicMock()
    mock.generate.return_value = "Test response"
    mock.get_rating.return_value = 0.8
    mock.get_structured_output.return_value = {"test": "response"}
    return mock

# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def assert_embeddings_similar(
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        threshold: float = 0.9
    ) -> None:
        """Assert two embeddings are similar"""
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        )
        assert similarity.item() > threshold
    
    @staticmethod
    def create_test_file(
        temp_dir: Path,
        content: str,
        filename: str
    ) -> Path:
        """Create a test file with content"""
        file_path = temp_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    @staticmethod
    def load_test_data(name: str) -> Dict[str, Any]:
        """Load test data from JSON file"""
        data_file = TestConfig.TEST_DATA_DIR / f"{name}.json"
        with open(data_file, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def create_random_text(length: int = 100) -> str:
        """Create random text for testing"""
        words = ["test", "risk", "analysis", "document", "content", 
                "data", "system", "process", "result", "value"]
        return " ".join(np.random.choice(words, size=length))
    
    @staticmethod
    def compare_outputs(output1: Any, output2: Any, tolerance: float = 1e-6) -> bool:
        """Compare two outputs with tolerance"""
        if isinstance(output1, (int, float)) and isinstance(output2, (int, float)):
            return abs(output1 - output2) < tolerance
        elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            return torch.allclose(output1, output2, atol=tolerance)
        elif isinstance(output1, dict) and isinstance(output2, dict):
            return all(TestUtils.compare_outputs(v1, v2, tolerance) 
                      for (v1, v2) in zip(output1.values(), output2.values()))
        return output1 == output2

# Mock classes for testing
class MockProcessor:
    """Mock document processor"""
    
    async def process_file(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "content": "Processed content",
            "metadata": metadata or {},
            "tables": [],
            "images": [],
            "file_hash": "test_hash"
        }

class MockEmbedder:
    """Mock embeddings generator"""
    
    async def encode(self, texts: List[str]) -> torch.Tensor:
        return torch.randn(len(texts), 768)
    
    async def encode_query(self, query: str) -> torch.Tensor:
        return torch.randn(768)

class MockVectorStore:
    """Mock vector store"""
    
    async def add(
        self,
        embeddings: torch.Tensor,
        documents: List[Dict[str, Any]]
    ) -> None:
        pass
    
    async def search(
        self,
        query_embedding: torch.Tensor,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        return [
            {
                "content": "Test result",
                "score": 0.9,
                "metadata": {"source": "test"}
            }
            for _ in range(k)
        ]

class MockSummarizer:
    """Mock document summarizer"""
    
    async def build_summary_tree(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "summary": "Test summary",
            "children": [],
            "metadata": metadata or {}
        }

# Test decorators
def requires_gpu(func):
    """Decorator to skip test if GPU not available"""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Test requires GPU"
    )(func)

def slow_test(func):
    """Decorator to mark slow tests"""
    return pytest.mark.slow(func)

def with_timeout(func):
    """Decorator to add timeout to async tests"""
    return pytest.mark.asyncio(pytest.mark.timeout(TestConfig.TEST_TIMEOUT)(func))

# Test base classes
class BaseTestCase:
    """Base class for test cases"""
    
    @classmethod
    def setup_class(cls):
        """Set up test class"""
        cls.test_dir = TestConfig.TEMP_DIR / cls.__name__
        cls.test_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def teardown_class(cls):
        """Clean up after test class"""
        shutil.rmtree(cls.test_dir)
    
    def setup_method(self):
        """Set up test method"""
        self.method_dir = self.test_dir / self._testMethodName
        self.method_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up after test method"""
        shutil.rmtree(self.method_dir)

class AsyncTestCase(BaseTestCase):
    """Base class for async test cases"""
    
    @pytest.fixture(autouse=True)
    def inject_event_loop(self, event_loop):
        """Inject event loop into test case"""
        self.loop = event_loop
        
    async def run_async(self, coro):
        """Run coroutine with timeout"""
        return await asyncio.wait_for(coro, timeout=TestConfig.TEST_TIMEOUT)

class MockTestCase(BaseTestCase):
    """Base class for tests with mocked dependencies"""
    
    def setup_method(self):
        """Set up test method with mocked dependencies"""
        super().setup_method()
        
        # Mock common dependencies
        self.mock_llm = MagicMock()
        self.mock_embedder = MockEmbedder()
        self.mock_processor = MockProcessor()
        self.mock_vector_store = MockVectorStore()
        self.mock_summarizer = MockSummarizer()
        
        # Set up patches
        self.patches = [
            patch('risk_rag_system.llm.MistralRouter', return_value=self.mock_llm),
            patch('risk_rag_system.indexing.BGEEmbedder', return_value=self.mock_embedder),
            patch('risk_rag_system.input_processing.DocumentProcessor', return_value=self.mock_processor),
            patch('risk_rag_system.indexing.VectorStore', return_value=self.mock_vector_store),
            patch('risk_rag_system.indexing.RaptorSummarizer', return_value=self.mock_summarizer)
        ]
        
        # Start patches
        for p in self.patches:
            p.start()
    
    def teardown_method(self):
        """Clean up mocked dependencies"""
        # Stop patches
        for p in self.patches:
            p.stop()
        
        super().teardown_method()

class IntegrationTestCase(AsyncTestCase):
    """Base class for integration tests"""
    
    def setup_method(self):
        """Set up integration test environment"""
        super().setup_method()
        
        # Set up test data directory
        self.test_data_dir = self.method_dir / "test_data"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up model cache
        self.model_cache_dir = self.method_dir / "model_cache"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up vector store
        self.vector_store_dir = self.method_dir / "vector_store"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure test environment
        self.env_vars = {
            "MODEL_CACHE_DIR": str(self.model_cache_dir),
            "VECTOR_STORE_DIR": str(self.vector_store_dir),
            "TEST_MODE": "true"
        }
        
        # Start test environment
        with patch.dict('os.environ', self.env_vars):
            self._start_environment()
    
    def teardown_method(self):
        """Clean up integration test environment"""
        self._stop_environment()
        super().teardown_method()
    
    def _start_environment(self):
        """Start integration test environment"""
        # Initialize necessary components
        logger.info("Starting test environment")
    
    def _stop_environment(self):
        """Stop integration test environment"""
        # Cleanup components
        logger.info("Stopping test environment")

# Test mixins
class EmbeddingTestMixin:
    """Mixin for embedding-related tests"""
    
    def assert_valid_embeddings(
        self,
        embeddings: torch.Tensor,
        expected_dim: int,
        batch_size: Optional[int] = None
    ):
        """Assert embeddings have valid format"""
        assert isinstance(embeddings, torch.Tensor)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[1] == expected_dim
        if batch_size:
            assert embeddings.shape[0] == batch_size
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
        assert torch.all(torch.abs(embeddings) <= 100)  # Reasonable magnitude

class DocumentTestMixin:
    """Mixin for document-related tests"""
    
    def create_test_document(
        self,
        content: str,
        doc_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create test document with metadata"""
        return {
            "content": content,
            "type": doc_type,
            "metadata": metadata or {},
            "id": f"test_doc_{hash(content)}"
        }
    
    def assert_valid_document(self, document: Dict[str, Any]):
        """Assert document has valid format"""
        assert "content" in document
        assert "type" in document
        assert "metadata" in document
        assert isinstance(document["metadata"], dict)
        if "id" in document:
            assert isinstance(document["id"], str)

class NodeTestMixin:
    """Mixin for node-related tests"""
    
    async def setup_test_node(self, node_class, config=None):
        """Set up a test node instance"""
        node = node_class(node_id="test_node", config=config)
        await node._initialize_node()
        return node
    
    def assert_valid_node_output(self, output):
        """Assert node output has valid format"""
        assert hasattr(output, "content")
        assert hasattr(output, "confidence_score")
        assert isinstance(output.confidence_score, float)
        assert 0 <= output.confidence_score <= 1
        if hasattr(output, "metadata"):
            assert isinstance(output.metadata, dict)

# Context managers for testing
class TempModelContext:
    """Context manager for temporary model files"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
    
    async def __aenter__(self):
        # Set up temporary model
        return self.model_path
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Clean up temporary model
        if self.model_path.exists():
            self.model_path.unlink()

class MockAPIContext:
    """Context manager for mocked API calls"""
    
    def __init__(self, responses: Dict[str, Any]):
        self.responses = responses
        self.patches = []
    
    def __enter__(self):
        # Set up API mocks
        for endpoint, response in self.responses.items():
            mock = patch(f'aiohttp.ClientSession.{endpoint}')
            mock.return_value.__aenter__.return_value.json.return_value = response
            self.patches.append(mock)
            mock.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up patches
        for p in self.patches:
            p.stop()