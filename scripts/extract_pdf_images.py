#!/usr/bin/env python3
"""
Extract Images from PDF using PyMuPDF
"""

import fitz  # PyMuPDF
import json
from pathlib import Path
from datetime import datetime

def extract_pdf_images(pdf_path: str):
    """Extract images from PDF and save them"""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    print(f"🔍 Processing PDF: {pdf_path.name}")
    
    # Create output directory
    output_dir = Path("data/raw/pdf/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        
        extraction_data = {
            'metadata': {
                'filename': pdf_path.name,
                'file_path': str(pdf_path),
                'file_size_bytes': pdf_path.stat().st_size,
                'file_size_mb': round(pdf_path.stat().st_size / (1024 * 1024), 2),
                'extracted_at': datetime.now().isoformat(),
                'page_count': len(doc),
                'extraction_method': 'pymupdf_images'
            },
            'pages': [],
            'statistics': {
                'total_pages': len(doc),
                'total_images': 0,
                'total_words': 0
            }
        }
        
        total_images = 0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Get page dimensions
            rect = page.rect
            
            # Extract images from page
            image_list = page.get_images()
            
            page_data = {
                'page_number': page_num + 1,
                'dimensions': {
                    'width': rect.width,
                    'height': rect.height
                },
                'images': [],
                'text': page.get_text(),
                'word_count': len(page.get_text().split())
            }
            
            # Process each image
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Save image
                        img_filename = f"{pdf_path.stem}_page_{page_num + 1}_img_{img_index + 1}.png"
                        img_path = output_dir / img_filename
                        pix.save(str(img_path))
                        
                        img_data = {
                            'image_number': img_index + 1,
                            'filename': img_filename,
                            'file_path': str(img_path),
                            'width': pix.width,
                            'height': pix.height,
                            'colorspace': pix.colorspace.name if pix.colorspace else 'unknown'
                        }
                        
                        page_data['images'].append(img_data)
                        total_images += 1
                        
                        print(f"   📷 Saved image: {img_filename} ({pix.width}x{pix.height})")
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    print(f"   ⚠️ Error processing image {img_index + 1} on page {page_num + 1}: {e}")
            
            extraction_data['pages'].append(page_data)
            extraction_data['statistics']['total_words'] += page_data['word_count']
            
            print(f"   📄 Page {page_num + 1}: {len(page_data['images'])} images, {page_data['word_count']} words")
        
        extraction_data['statistics']['total_images'] = total_images
        
        # Save extraction data
        output_file = Path("data/raw/pdf/EGYPT & Tourist Destinations_images.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Extraction completed!")
        print(f"   📊 Total pages: {extraction_data['statistics']['total_pages']}")
        print(f"   🖼️ Total images: {extraction_data['statistics']['total_images']}")
        print(f"   📝 Total words: {extraction_data['statistics']['total_words']}")
        print(f"   💾 Data saved to: {output_file}")
        print(f"   🖼️ Images saved to: {output_dir}")
        
        doc.close()
        return extraction_data
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    pdf_path = "data/raw/pdf/EGYPT & Tourist Destinations.pdf"
    extract_pdf_images(pdf_path) 