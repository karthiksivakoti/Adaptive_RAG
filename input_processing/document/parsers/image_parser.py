# risk_rag_system/input_processing/document/parsers/image_parser.py

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from loguru import logger
from pydantic import BaseModel
import io
import numpy as np
from datetime import datetime

class ImageParserConfig(BaseModel):
    """Configuration for image parsing"""
    perform_ocr: bool = True
    ocr_lang: str = "eng"
    enhance_image: bool = True
    max_dimension: Optional[int] = 4096
    min_confidence: float = 60.0
    enable_preprocessing: bool = True
    output_format: str = "png"
    dpi: int = 300
    extract_metadata: bool = True
    supported_formats: List[str] = [
        "jpg", "jpeg", "png", "tiff", "bmp", "gif"
    ]

class ImageParser:
    """Parser for image documents"""
    
    def __init__(self, config: Optional[ImageParserConfig] = None):
        self.config = config or ImageParserConfig()
        logger.info("Initialized ImageParser")

    async def parse(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Parse image document"""
        try:
            logger.info(f"Parsing image: {file_path}")
            
            # Validate format
            if not self._validate_format(file_path):
                raise ValueError(f"Unsupported image format: {file_path.suffix}")
            
            # Load image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                # Resize if needed
                if self.config.max_dimension:
                    img = self._resize_image(img)
                
                # Preprocess image if enabled
                if self.config.enable_preprocessing:
                    processed_img = self._preprocess_image(img)
                else:
                    processed_img = img.copy()
                
                # Extract image properties
                img_props = self._get_image_properties(img)
                
                # Extract text content if OCR enabled
                content = ""
                if self.config.perform_ocr:
                    content = await self._perform_ocr(processed_img)
                
                # Save processed image
                output_buffer = io.BytesIO()
                processed_img.save(
                    output_buffer,
                    format=self.config.output_format.upper(),
                    dpi=(self.config.dpi, self.config.dpi)
                )
                
                # Prepare result
                result = {
                    "content": content,
                    "metadata": {
                        "source": str(file_path),
                        "type": "image",
                        **img_props,
                        **(metadata or {})
                    },
                    "images": [{
                        "content": output_buffer.getvalue(),
                        "format": self.config.output_format,
                        "size": processed_img.size,
                        "source": "processed_image"
                    }],
                    "tables": []  # Images don't have tables
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error parsing image {file_path}: {e}")
            raise

    def _validate_format(self, file_path: Path) -> bool:
        """Validate image format"""
        return file_path.suffix.lower().lstrip('.') in self.config.supported_formats

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        
        if max(width, height) <= self.config.max_dimension:
            return image
            
        # Calculate new dimensions
        if width > height:
            new_width = self.config.max_dimension
            new_height = int(height * (self.config.max_dimension / width))
        else:
            new_height = self.config.max_dimension
            new_width = int(width * (self.config.max_dimension / height))
            
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR results"""
        try:
            # Convert to grayscale for OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Apply noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Binarization using Otsu's method
            image = image.point(lambda x: 0 if x < 128 else 255, '1')
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image

    async def _perform_ocr(self, image: Image.Image) -> str:
        """Perform OCR on image"""
        try:
            # Get OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.config.ocr_lang,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text with confidence filtering
            text_parts = []
            for i, conf in enumerate(ocr_data['conf']):
                if conf > self.config.min_confidence:
                    text = ocr_data['text'][i].strip()
                    if text:
                        text_parts.append(text)
            
            return ' '.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error performing OCR: {e}")
            return ""

    def _get_image_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Extract image properties"""
        try:
            props = {
                "size": image.size,
                "mode": image.mode,
                "format": image.format,
                "dpi": image.info.get('dpi', (self.config.dpi, self.config.dpi))
            }
            
            # Extract EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                exif_data = {}
                
                # Common EXIF tags
                tags = {
                    271: 'make',          # Camera manufacturer
                    272: 'model',         # Camera model
                    306: 'datetime',      # Date and time
                    37377: 'shutter',     # Shutter speed
                    37378: 'aperture',    # Aperture
                    37379: 'brightness',  # Brightness
                    37380: 'exposure',    # Exposure bias
                    37383: 'metering',    # Metering mode
                    37385: 'flash',       # Flash
                    37386: 'focal',       # Focal length
                    41987: 'wb',          # White balance
                }
                
                for tag_id, tag_name in tags.items():
                    if tag_id in exif:
                        exif_data[tag_name] = str(exif[tag_id])
                
                if exif_data:
                    props['exif'] = exif_data
            
            return props
            
        except Exception as e:
            logger.error(f"Error getting image properties: {e}")
            return {
                "size": image.size,
                "mode": image.mode,
                "format": image.format
            }

    async def batch_process(
        self,
        image_files: List[Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple images"""
        results = []
        
        for file_path in image_files:
            try:
                result = await self.parse(file_path, metadata)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        return results

    def get_state(self) -> Dict[str, Any]:
        """Get current state of parser"""
        return {
            "config": self.config.dict(),
            "supported_formats": self.config.supported_formats,
            "ocr_enabled": self.config.perform_ocr,
            "preprocessing_enabled": self.config.enable_preprocessing
        }