# risk_rag_system/tests/test_input_processing/test_document_processor.py

import pytest
from pathlib import Path
import numpy as np
from PIL import Image
import io
import pandas as pd
from typing import Dict, Any

from tests.test_base import (
    MockTestCase,
    DocumentTestMixin,
    TestUtils,
    slow_test,
    with_timeout
)
from input_processing.document.processor import (
    DocumentProcessor,
    ProcessorConfig,
    ProcessedDocument
)

class TestDocumentProcessor(MockTestCase, DocumentTestMixin):
    """Tests for DocumentProcessor"""
    
    async def setup_method(self):
        """Set up test method"""
        await super().setup_method()
        self.config = ProcessorConfig(
            extract_tables=True,
            ocr_enabled=True,
            preserve_formatting=True
        )
        self.processor = DocumentProcessor(self.config)
        
        # Create test directory structure
        self.test_files_dir = self.method_dir / "test_files"
        self.test_files_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test files
        await self._create_test_files()
    
    async def _create_test_files(self):
        """Create various test files"""
        # Text file
        self.text_file = TestUtils.create_test_file(
            self.test_files_dir,
            "Test content for text file.\nMultiple lines of text.",
            "test.txt"
        )
        
        # PDF file (mock)
        self.pdf_content = b"%PDF-1.4\nMock PDF content"
        self.pdf_file = self.test_files_dir / "test.pdf"
        self.pdf_file.write_bytes(self.pdf_content)
        
        # Excel file
        self.excel_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        self.excel_file = self.test_files_dir / "test.xlsx"
        self.excel_data.to_excel(self.excel_file, index=False)
        
        # Image file
        self.image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.image = Image.fromarray(self.image_array)
        self.image_file = self.test_files_dir / "test.png"
        self.image.save(self.image_file)
    
    @pytest.mark.asyncio
    async def test_process_text_file(self):
        """Test processing text file"""
        result = await self.processor.process_file(
            self.text_file,
            metadata={"source": "test"}
        )
        
        # Verify result
        assert isinstance(result, ProcessedDocument)
        assert result.content.strip() == "Test content for text file.\nMultiple lines of text."
        assert result.metadata["source"] == "test"
        assert result.metadata["file_type"] == ".txt"
        assert len(result.tables) == 0
        assert len(result.images) == 0
    
    @pytest.mark.asyncio
    async def test_process_excel_file(self):
        """Test processing Excel file"""
        result = await self.processor.process_file(self.excel_file)
        
        # Verify result
        assert isinstance(result, ProcessedDocument)
        assert len(result.tables) > 0
        
        # Verify table content
        table = result.tables[0]
        assert table["name"] == "Sheet1"
        assert len(table["data"]) == 3
        assert table["columns"] == ['A', 'B']
    
    @slow_test
    @pytest.mark.asyncio
    async def test_process_image_file(self):
        """Test processing image file with OCR"""
        # Mock OCR response
        self.mock_processor.image_to_string.return_value = "OCR extracted text"
        
        result = await self.processor.process_file(self.image_file)
        
        # Verify result
        assert isinstance(result, ProcessedDocument)
        assert "OCR extracted text" in result.content
        assert len(result.images) == 1
        assert result.images[0]["type"] == "png"
    
    @pytest.mark.asyncio
    async def test_process_pdf_with_tables(self):
        """Test processing PDF with table extraction"""
        result = await self.processor.process_file(self.pdf_file)
        
        # Verify result
        assert isinstance(result, ProcessedDocument)
        assert result.metadata["file_type"] == ".pdf"
        
        # If tables were found
        if result.tables:
            for table in result.tables:
                assert "data" in table
                assert "num_rows" in table
                assert "num_cols" in table
    
    @pytest.mark.asyncio
    async def test_file_validation(self):
        """Test file validation"""
        # Test invalid file
        invalid_file = self.test_files_dir / "invalid.xyz"
        invalid_file.touch()
        
        with pytest.raises(ValueError):
            await self.processor.process_file(invalid_file)
        
        # Test file size limit
        large_file = self.test_files_dir / "large.txt"
        large_file.write_bytes(b"x" * (self.config.max_file_size_mb * 1024 * 1024 + 1))
        
        with pytest.raises(ValueError):
            await self.processor.process_file(large_file)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of multiple files"""
        files = [
            self.text_file,
            self.excel_file,
            self.image_file
        ]
        
        results = []
        for file in files:
            result = await self.processor.process_file(file)
            results.append(result)
        
        # Verify results
        assert len(results) == 3
        assert all(isinstance(r, ProcessedDocument) for r in results)
        
        # Verify different file types were handled correctly
        assert any(r.metadata["file_type"] == ".txt" for r in results)
        assert any(r.metadata["file_type"] == ".xlsx" for r in results)
        assert any(r.metadata["file_type"] == ".png" for r in results)
    
    @pytest.mark.asyncio
    async def test_extraction_options(self):
        """Test different extraction options"""
        # Test without table extraction
        no_tables_config = ProcessorConfig(extract_tables=False)
        no_tables_processor = DocumentProcessor(no_tables_config)
        
        result = await no_tables_processor.process_file(self.excel_file)
        assert len(result.tables) == 0
        
        # Test without OCR
        no_ocr_config = ProcessorConfig(ocr_enabled=False)
        no_ocr_processor = DocumentProcessor(no_ocr_config)
        
        result = await no_ocr_processor.process_file(self.image_file)
        assert not result.content  # No OCR text
    
    @pytest.mark.asyncio
    async def test_formatting_preservation(self):
        """Test formatting preservation"""
        # Create formatted text file
        formatted_content = """# Heading 1
        
        **Bold text**
        *Italic text*
        
        - List item 1
        - List item 2
        """
        formatted_file = TestUtils.create_test_file(
            self.test_files_dir,
            formatted_content,
            "formatted.txt"
        )
        
        # Test with formatting preserved
        preserve_config = ProcessorConfig(preserve_formatting=True)
        preserve_processor = DocumentProcessor(preserve_config)
        
        result = await preserve_processor.process_file(formatted_file)
        assert "#" in result.content  # Heading marker preserved
        assert "**" in result.content  # Bold markers preserved
        
        # Test without formatting preserved
        no_preserve_config = ProcessorConfig(preserve_formatting=False)
        no_preserve_processor = DocumentProcessor(no_preserve_config)
        
        result = await no_preserve_processor.process_file(formatted_file)
        assert "Heading 1" in result.content
        assert "Bold text" in result.content
        assert "**" not in result.content  # Markers removed