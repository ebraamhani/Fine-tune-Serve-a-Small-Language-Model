#!/usr/bin/env python3
"""
Manual PDF OCR Processing Script
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_collection.pdf_extractor import TourismPDFExtractor
import json

def main():
    """Process the PDF with OCR"""
    pdf_path = "data/raw/pdf/EGYPT & Tourist Destinations.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print("🔍 Processing PDF with OCR...")
    
    # Initialize extractor
    extractor = TourismPDFExtractor()
    
    try:
        # Extract with images and OCR
        result = extractor.extract_from_pdf(pdf_path, extract_images=True)
        
        if result and 'image_extraction' in result:
            image_data = result['image_extraction']
            print(f"✅ OCR completed!")
            print(f"   Pages processed: {image_data['statistics']['total_pages']}")
            print(f"   Words extracted: {image_data['statistics']['total_words_ocr']}")
            
            # Show sample OCR text
            if image_data['pages']:
                sample_page = image_data['pages'][0]
                print(f"\n📄 Sample OCR text from page 1:")
                print(f"   {sample_page['ocr_text'][:300]}...")
            
            # Save updated extraction
            output_file = Path("data/raw/pdf/EGYPT & Tourist Destinations_extraction.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Updated extraction saved to: {output_file}")
            
        else:
            print("❌ No image extraction data found")
            
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 