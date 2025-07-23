#!/usr/bin/env python3
"""
Data Collection Runner Script
Runs the complete data collection pipeline for the Tourist QA Assistant
"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_collection import TourismDataCollector

def main():
    """Main function to run data collection"""
    parser = argparse.ArgumentParser(description='Run data collection for Tourist QA Assistant')
    parser.add_argument('--output-dir', default='data/raw', help='Output directory for collected data')
    parser.add_argument('--include-travel-guides', action='store_true', default=True, 
                       help='Include travel guide websites in scraping')
    parser.add_argument('--extract-images', action='store_true', default=False,
                       help='Extract images from PDFs using OCR')
    parser.add_argument('--pdf-dir', help='Directory containing PDF files to process')
    parser.add_argument('--web-only', action='store_true', help='Only collect web data')
    parser.add_argument('--pdf-only', action='store_true', help='Only collect PDF data')
    parser.add_argument('--qa-only', action='store_true', help='Only generate Q&A pairs from existing data')
    
    args = parser.parse_args()
    
    print("🚀 Starting Data Collection for Tourist QA Assistant")
    print("=" * 60)
    
    # Initialize collector
    collector = TourismDataCollector(base_output_dir=args.output_dir)
    
    if args.qa_only:
        print("📝 Generating Q&A pairs from existing data...")
        qa_pairs = collector.create_qa_pairs()
        print(f"✅ Generated {len(qa_pairs)} Q&A pairs")
        return
    
    if args.web_only:
        print("🌐 Collecting web data only...")
        web_summary = collector.collect_web_data(args.include_travel_guides)
        print(f"✅ Web collection completed: {web_summary['statistics']['total_sources']} sources")
        return
    
    if args.pdf_only:
        print("📄 Collecting PDF data only...")
        pdf_summary = collector.collect_pdf_data(args.pdf_dir, args.extract_images)
        print(f"✅ PDF collection completed: {pdf_summary['statistics']['total_files']} files")
        return
    
    # Run full collection
    print("🔄 Running complete data collection pipeline...")
    print(f"Output directory: {args.output_dir}")
    print(f"Include travel guides: {args.include_travel_guides}")
    print(f"Extract images: {args.extract_images}")
    if args.pdf_dir:
        print(f"PDF directory: {args.pdf_dir}")
    
    try:
        summary = collector.run_full_collection(
            include_travel_guides=args.include_travel_guides,
            extract_images=args.extract_images
        )
        
        print("\n" + "=" * 60)
        print("🎉 Data Collection Completed Successfully!")
        print("=" * 60)
        
        # Print summary
        print(f"📊 Collection Statistics:")
        print(f"   • Total sources: {summary['total_statistics']['total_sources']}")
        print(f"   • Total PDF files: {summary['total_statistics']['total_pdf_files']}")
        print(f"   • Total words: {summary['total_statistics']['total_words']:,}")
        print(f"   • Total Q&A pairs: {summary['total_statistics']['total_qa_pairs']}")
        print(f"   • Duration: {summary['collection_session']['duration_minutes']:.1f} minutes")
        
        print(f"\n📁 Data saved to: {args.output_dir}")
        print(f"📋 Summary saved to: {args.output_dir}/metadata/overall_collection_summary.json")
        print(f"❓ Q&A pairs saved to: {args.output_dir}/qa_pairs.json")
        
        # Print Q&A categories
        if 'qa_generation' in summary:
            print(f"\n📂 Q&A Categories:")
            for category in summary['qa_generation']['categories']:
                print(f"   • {category}")
        
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 