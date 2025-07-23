"""
PDF Extractor for Tourist QA Assistant
Extracts text and layout information from tourism-related PDF documents
"""

import fitz  # PyMuPDF
import pdfplumber
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import re
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TourismPDFExtractor:
    """PDF extractor for tourism-related documents"""
    
    def __init__(self, output_dir: str = "data/raw/pdf"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different extraction methods
        (self.output_dir / "text").mkdir(exist_ok=True)
        (self.output_dir / "layout").mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        
    def extract_from_pdf(self, pdf_path: str, extract_images: bool = False) -> Dict[str, Any]:
        """Extract text and layout from a PDF file"""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Create metadata
        metadata = self._create_metadata(pdf_path)
        
        # Extract text using PyMuPDF
        text_data = self._extract_text_pymupdf(pdf_path)
        
        # Extract layout using pdfplumber
        layout_data = self._extract_layout_pdfplumber(pdf_path)
        
        # Extract images if requested
        image_data = None
        if extract_images:
            image_data = self._extract_images_pdf2image(pdf_path)
        
        # Combine all data
        extracted_data = {
            'metadata': metadata,
            'text_extraction': text_data,
            'layout_extraction': layout_data
        }
        
        if image_data:
            extracted_data['image_extraction'] = image_data
        
        # Save extracted data
        self._save_extracted_data(extracted_data, pdf_path)
        
        return extracted_data
    
    def _create_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Create metadata for the PDF file"""
        try:
            # Get file info
            stat = pdf_path.stat()
            
            # Get PDF info using PyMuPDF
            doc = fitz.open(pdf_path)
            
            metadata = {
                'filename': pdf_path.name,
                'file_path': str(pdf_path),
                'file_size_bytes': stat.st_size,
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'extracted_at': datetime.now().isoformat(),
                'pdf_info': {
                    'page_count': len(doc),
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', ''),
                    'creator': doc.metadata.get('creator', ''),
                    'producer': doc.metadata.get('producer', ''),
                    'creation_date': doc.metadata.get('creationDate', ''),
                    'modification_date': doc.metadata.get('modDate', '')
                },
                'extraction_methods': ['pymupdf', 'pdfplumber']
            }
            
            doc.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error creating metadata for {pdf_path}: {e}")
            return {
                'filename': pdf_path.name,
                'file_path': str(pdf_path),
                'extracted_at': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _extract_text_pymupdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            
            text_data = {
                'method': 'pymupdf',
                'pages': [],
                'full_text': '',
                'statistics': {
                    'total_pages': len(doc),
                    'total_words': 0,
                    'total_characters': 0
                }
            }
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                
                # Get page dimensions
                rect = page.rect
                
                page_data = {
                    'page_number': page_num + 1,
                    'text': text,
                    'dimensions': {
                        'width': rect.width,
                        'height': rect.height
                    },
                    'word_count': len(text.split()),
                    'char_count': len(text)
                }
                
                text_data['pages'].append(page_data)
                text_data['full_text'] += text + '\n'
                text_data['statistics']['total_words'] += page_data['word_count']
                text_data['statistics']['total_characters'] += page_data['char_count']
            
            doc.close()
            return text_data
            
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF from {pdf_path}: {e}")
            return {
                'method': 'pymupdf',
                'error': str(e),
                'pages': [],
                'full_text': '',
                'statistics': {'total_pages': 0, 'total_words': 0, 'total_characters': 0}
            }
    
    def _extract_layout_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract layout information using pdfplumber"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                
                layout_data = {
                    'method': 'pdfplumber',
                    'pages': [],
                    'tables': [],
                    'statistics': {
                        'total_pages': len(pdf.pages),
                        'total_tables': 0,
                        'total_images': 0
                    }
                }
                
                for page_num, page in enumerate(pdf.pages):
                    page_data = {
                        'page_number': page_num + 1,
                        'width': page.width,
                        'height': page.height,
                        'text_blocks': [],
                        'tables': [],
                        'images': []
                    }
                    
                    # Extract text blocks with positioning
                    text_blocks = page.extract_text_blocks()
                    for block in text_blocks:
                        page_data['text_blocks'].append({
                            'text': block['text'],
                            'x0': block['x0'],
                            'y0': block['y0'],
                            'x1': block['x1'],
                            'y1': block['y1'],
                            'font': block.get('font', ''),
                            'size': block.get('size', 0)
                        })
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table_num, table in enumerate(tables):
                        table_data = {
                            'page': page_num + 1,
                            'table_number': table_num + 1,
                            'data': table,
                            'rows': len(table),
                            'columns': len(table[0]) if table else 0
                        }
                        page_data['tables'].append(table_data)
                        layout_data['tables'].append(table_data)
                        layout_data['statistics']['total_tables'] += 1
                    
                    # Extract images
                    images = page.images
                    for img_num, img in enumerate(images):
                        img_data = {
                            'page': page_num + 1,
                            'image_number': img_num + 1,
                            'x0': img['x0'],
                            'y0': img['y0'],
                            'x1': img['x1'],
                            'y1': img['y1'],
                            'width': img['width'],
                            'height': img['height']
                        }
                        page_data['images'].append(img_data)
                        layout_data['statistics']['total_images'] += 1
                    
                    layout_data['pages'].append(page_data)
                
                return layout_data
                
        except Exception as e:
            logger.error(f"Error extracting layout with pdfplumber from {pdf_path}: {e}")
            return {
                'method': 'pdfplumber',
                'error': str(e),
                'pages': [],
                'tables': [],
                'statistics': {'total_pages': 0, 'total_tables': 0, 'total_images': 0}
            }
    
    def _extract_images_pdf2image(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract images from PDF using pdf2image and OCR"""
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=200)
            
            image_data = {
                'method': 'pdf2image',
                'pages': [],
                'statistics': {
                    'total_pages': len(images),
                    'total_words_ocr': 0
                }
            }
            
            for page_num, image in enumerate(images):
                # Save image
                image_filename = f"{pdf_path.stem}_page_{page_num + 1}.png"
                image_path = self.output_dir / "images" / image_filename
                image.save(image_path, 'PNG')
                
                # Perform OCR
                try:
                    ocr_text = pytesseract.image_to_string(image)
                    ocr_words = len(ocr_text.split())
                except Exception as ocr_error:
                    logger.warning(f"OCR failed for page {page_num + 1}: {ocr_error}")
                    ocr_text = ""
                    ocr_words = 0
                
                page_data = {
                    'page_number': page_num + 1,
                    'image_filename': image_filename,
                    'image_path': str(image_path),
                    'ocr_text': ocr_text,
                    'ocr_word_count': ocr_words,
                    'image_size': image.size
                }
                
                image_data['pages'].append(page_data)
                image_data['statistics']['total_words_ocr'] += ocr_words
            
            return image_data
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            return {
                'method': 'pdf2image',
                'error': str(e),
                'pages': [],
                'statistics': {'total_pages': 0, 'total_words_ocr': 0}
            }
    
    def _save_extracted_data(self, data: Dict[str, Any], pdf_path: Path) -> None:
        """Save extracted data to JSON files"""
        try:
            # Create base filename
            base_filename = pdf_path.stem
            
            # Save main extraction data
            main_file = self.output_dir / f"{base_filename}_extraction.json"
            with open(main_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save text-only data
            if 'text_extraction' in data and 'full_text' in data['text_extraction']:
                text_file = self.output_dir / "text" / f"{base_filename}_text.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(data['text_extraction']['full_text'])
            
            # Save layout data
            if 'layout_extraction' in data:
                layout_file = self.output_dir / "layout" / f"{base_filename}_layout.json"
                with open(layout_file, 'w', encoding='utf-8') as f:
                    json.dump(data['layout_extraction'], f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved extraction data: {main_file}")
            
        except Exception as e:
            logger.error(f"Error saving extracted data for {pdf_path}: {e}")
    
    def process_pdf_directory(self, pdf_dir: str, extract_images: bool = False) -> Dict[str, Any]:
        """Process all PDF files in a directory"""
        pdf_dir = Path(pdf_dir)
        
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        
        logger.info(f"Processing PDF directory: {pdf_dir}")
        
        summary = {
            'processing_session': {
                'started_at': datetime.now().isoformat(),
                'pdf_directory': str(pdf_dir),
                'total_files': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'total_words': 0,
                'total_tables': 0
            },
            'files': []
        }
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing: {pdf_file.name}")
                
                # Extract data
                extracted_data = self.extract_from_pdf(pdf_file, extract_images)
                
                # Update summary
                summary['processing_session']['total_files'] += 1
                summary['processing_session']['successful_extractions'] += 1
                
                # Add statistics
                if 'text_extraction' in extracted_data:
                    text_stats = extracted_data['text_extraction'].get('statistics', {})
                    summary['processing_session']['total_words'] += text_stats.get('total_words', 0)
                
                if 'layout_extraction' in extracted_data:
                    layout_stats = extracted_data['layout_extraction'].get('statistics', {})
                    summary['processing_session']['total_tables'] += layout_stats.get('total_tables', 0)
                
                # Add file info
                summary['files'].append({
                    'filename': pdf_file.name,
                    'extraction_successful': True,
                    'word_count': extracted_data.get('text_extraction', {}).get('statistics', {}).get('total_words', 0),
                    'table_count': extracted_data.get('layout_extraction', {}).get('statistics', {}).get('total_tables', 0),
                    'extracted_at': extracted_data['metadata']['extracted_at']
                })
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                summary['processing_session']['total_files'] += 1
                summary['processing_session']['failed_extractions'] += 1
                
                summary['files'].append({
                    'filename': pdf_file.name,
                    'extraction_successful': False,
                    'error': str(e),
                    'extracted_at': datetime.now().isoformat()
                })
        
        # Save summary
        summary_file = self.output_dir / "pdf_extraction_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"PDF processing completed! Summary saved: {summary_file}")
        return summary

def download_sample_pdfs():
    """Download sample tourism-related PDFs for testing"""
    import requests
    
    sample_pdfs = [
        {
            'url': 'https://www.egypt.travel/media/egypt-travel-guide.pdf',
            'filename': 'egypt_travel_guide.pdf'
        },
        {
            'url': 'https://www.visa2egypt.gov.eg/media/visa-information.pdf',
            'filename': 'visa_information.pdf'
        }
    ]
    
    pdf_dir = Path("data/raw/pdf_samples")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    for pdf_info in sample_pdfs:
        try:
            response = requests.get(pdf_info['url'], timeout=30)
            if response.status_code == 200:
                pdf_path = pdf_dir / pdf_info['filename']
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded: {pdf_path}")
            else:
                logger.warning(f"Failed to download {pdf_info['url']}")
        except Exception as e:
            logger.error(f"Error downloading {pdf_info['url']}: {e}")

if __name__ == "__main__":
    # Example usage
    extractor = TourismPDFExtractor()
    
    # Process a single PDF
    # extractor.extract_from_pdf("path/to/your/pdf/file.pdf")
    
    # Process a directory of PDFs
    # summary = extractor.process_pdf_directory("data/raw/pdf_samples")
    # print(json.dumps(summary, indent=2)) 