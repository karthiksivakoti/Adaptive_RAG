# risk_rag_system/input_processing/document/parsers/pdf_parser.py

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
from loguru import logger
from pydantic import BaseModel
import pytesseract
from concurrent.futures import ThreadPoolExecutor

class PDFParserConfig(BaseModel):
    """Configuration for PDF parsing"""
    extract_images: bool = True
    perform_ocr: bool = True
    min_image_size: int = 100  # minimum pixel dimension
    max_image_size: int = 4000  # maximum pixel dimension
    dpi: int = 300
    thread_pool_size: int = 4
    image_quality: int = 95
    ocr_lang: str = "eng"
    extract_tables: bool = True

class PDFParser:
    """Parser for PDF documents"""
    
    def __init__(self, config: Optional[PDFParserConfig] = None):
        self.config = config or PDFParserConfig()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        logger.info("Initialized PDFParser")

    async def parse(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Parse PDF document"""
        try:
            logger.info(f"Parsing PDF: {file_path}")
            doc = fitz.open(str(file_path))
            
            # Extract text and elements from all pages
            text_content = []
            images = []
            tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = await self._extract_page_text(page)
                text_content.append(text)
                
                # Extract images if enabled
                if self.config.extract_images:
                    page_images = await self._extract_page_images(page, page_num)
                    images.extend(page_images)
                
                # Extract tables if enabled
                if self.config.extract_tables:
                    page_tables = await self._extract_page_tables(page)
                    tables.extend(page_tables)
            
            # Process images with OCR if enabled
            if self.config.perform_ocr and images:
                ocr_text = await self._process_ocr(images)
                text_content.append(ocr_text)
            
            # Prepare result
            result = {
                "content": "\n\n".join(text_content),
                "metadata": {
                    "source": str(file_path),
                    "type": "pdf",
                    "num_pages": len(doc),
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "creation_date": doc.metadata.get("creationDate", ""),
                    "modification_date": doc.metadata.get("modDate", ""),
                    **(metadata or {})
                },
                "images": images,
                "tables": tables
            }
            
            doc.close()
            return result
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise

    async def _extract_page_text(
        self,
        page: fitz.Page
    ) -> str:
        """Extract text from PDF page"""
        try:
            # Get text blocks with formatting information
            blocks = page.get_text("dict")["blocks"]
            text_content = []
            
            for block in blocks:
                if block["type"] == 0:  # Regular text block
                    for line in block["lines"]:
                        line_text = []
                        for span in line["spans"]:
                            # Check for formatting
                            text = span["text"]
                            if span["flags"] & 2**3:  # Bold
                                text = f"**{text}**"
                            if span["flags"] & 2**1:  # Italic
                                text = f"*{text}*"
                            line_text.append(text)
                        text_content.append(" ".join(line_text))
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error extracting page text: {e}")
            return ""

    async def _extract_page_images(
        self,
        page: fitz.Page,
        page_num: int
    ) -> List[Dict[str, Any]]:
        """Extract images from PDF page"""
        try:
            images = []
            image_list = page.get_images(full=True)
            
            for img_idx, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = page.parent.extract_image(xref)
                    
                    if base_image:
                        image_data = {
                            "content": base_image["image"],
                            "size": (base_image.get("width", 0), base_image.get("height", 0)),
                            "format": base_image["ext"],
                            "page": page_num,
                            "index": img_idx,
                            "dpi": base_image.get("dpi", (self.config.dpi, self.config.dpi))
                        }
                        
                        # Filter by size
                        if (self.config.min_image_size <= image_data["size"][0] <= self.config.max_image_size and
                            self.config.min_image_size <= image_data["size"][1] <= self.config.max_image_size):
                            images.append(image_data)
                            
                except Exception as img_error:
                    logger.warning(f"Error extracting image {img_idx} from page {page_num}: {img_error}")
                    continue
            
            return images
            
        except Exception as e:
            logger.error(f"Error extracting page images: {e}")
            return []

    async def _extract_page_tables(
        self,
        page: fitz.Page
    ) -> List[Dict[str, Any]]:
        """Extract tables from PDF page"""
        try:
            tables = []
            # Find table-like structures based on text positioning
            words = page.get_text("words")
            if not words:
                return tables
            
            # Group words into potential table cells
            cells = self._group_words_into_cells(words)
            
            # Identify table structures
            table_structures = self._identify_tables(cells)
            
            # Convert to structured tables
            for table_idx, structure in enumerate(table_structures):
                table_data = self._convert_structure_to_table(structure)
                if table_data:
                    tables.append({
                        "data": table_data,
                        "num_rows": len(table_data),
                        "num_cols": len(table_data[0]) if table_data else 0,
                        "page": page.number,
                        "index": table_idx
                    })
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting page tables: {e}")
            return []

    def _group_words_into_cells(
        self,
        words: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """Group words into potential table cells"""
        cells = []
        current_line = []
        current_y = words[0][3] if words else 0
        
        for word in words:
            # Check if word is on new line
            if abs(word[3] - current_y) > 5:  # 5 point threshold
                if current_line:
                    cells.append(self._process_line(current_line))
                current_line = []
                current_y = word[3]
            current_line.append(word)
        
        # Add last line
        if current_line:
            cells.append(self._process_line(current_line))
        
        return cells

    def _process_line(
        self,
        line: List[List[float]]
    ) -> Dict[str, Any]:
        """Process line of words into cell structure"""
        return {
            "text": " ".join(word[4] for word in line),
            "bbox": (
                min(word[0] for word in line),  # left
                min(word[1] for word in line),  # top
                max(word[2] for word in line),  # right
                max(word[3] for word in line)   # bottom
            )
        }

    def _identify_tables(
        self,
        cells: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Identify table structures from cells"""
        tables = []
        current_table = []
        
        for i, cell in enumerate(cells):
            # Check if cell might be part of a table
            if i > 0 and self._is_table_row(cell, cells[i-1]):
                current_table.append(cell)
            else:
                if len(current_table) >= 3:  # Minimum 3 rows for a table
                    tables.append(current_table)
                current_table = [cell]
        
        # Add last table if exists
        if len(current_table) >= 3:
            tables.append(current_table)
        
        return tables

    def _is_table_row(
        self,
        cell1: Dict[str, Any],
        cell2: Dict[str, Any]
    ) -> bool:
        """Check if two cells might be part of same table"""
        # Check vertical spacing
        spacing = abs(cell1["bbox"][1] - cell2["bbox"][3])
        return spacing < 20  # 20 point threshold

    def _convert_structure_to_table(
        self,
        structure: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """Convert table structure to 2D array"""
        if not structure:
            return []
        
        # Find column boundaries
        x_positions = set()
        for cell in structure:
            x_positions.add(cell["bbox"][0])  # left
            x_positions.add(cell["bbox"][2])  # right
        x_positions = sorted(list(x_positions))
        
        # Create table grid
        table = []
        for cell in structure:
            row = []
            col_start = x_positions.index(cell["bbox"][0])
            col_end = x_positions.index(cell["bbox"][2])
            row.extend([""] * col_start)
            row.append(cell["text"])
            row.extend([""] * (len(x_positions) - col_end - 1))
            table.append(row)
        
        return table

    async def _process_ocr(
        self,
        images: List[Dict[str, Any]]
    ) -> str:
        """Process OCR on extracted images"""
        try:
            ocr_results = []
            
            for img_data in images:
                # Convert image bytes to PIL Image
                image = Image.open(io.BytesIO(img_data["content"]))
                
                # Perform OCR
                text = pytesseract.image_to_string(
                    image,
                    lang=self.config.ocr_lang
                )
                
                if text.strip():
                    ocr_results.append(text)
            
            return "\n\n".join(ocr_results)
            
        except Exception as e:
            logger.error(f"Error processing OCR: {e}")
            return ""