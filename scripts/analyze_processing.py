#!/usr/bin/env python3
"""
Analyze Data Processing Results
"""

import json
from pathlib import Path

def analyze_processing_results():
    """Analyze the data processing results"""
    
    # Load processing summary
    summary_file = Path("data/processed/metadata/processing_summary.json")
    
    if not summary_file.exists():
        print("❌ Processing summary not found")
        return
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print("📊 Data Processing Results Analysis")
    print("=" * 60)
    
    # Processing session
    session = summary['processing_session']
    print(f"🔄 Processing Session:")
    print(f"   • Original pairs: {session['original_count']}")
    print(f"   • Processed pairs: {session['processed_count']}")
    print(f"   • Removed pairs: {session['removed_count']}")
    print(f"   • Retention rate: {session['retention_rate']:.1%}")
    
    # Quality metrics
    quality = summary['quality_metrics']
    print(f"\n📈 Quality Metrics:")
    print(f"   • Avg question length: {quality['avg_question_length']:.1f} chars")
    print(f"   • Avg answer length: {quality['avg_answer_length']:.1f} chars")
    print(f"   • Avg quality score: {quality['avg_quality_score']:.3f}")
    
    # Language distribution
    lang_dist = summary['language_distribution']
    print(f"\n🌐 Language Distribution:")
    for lang, count in lang_dist.items():
        print(f"   • {lang.upper()}: {count} pairs")
    
    # Content type distribution
    content_dist = summary['content_type_distribution']
    print(f"\n📚 Content Type Distribution:")
    for content_type, count in content_dist.items():
        print(f"   • {content_type}: {count} pairs")
    
    # Category distribution
    category_dist = summary['category_distribution']
    print(f"\n📂 Category Distribution:")
    for category, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {category}: {count} pairs")
    
    # Source type distribution
    source_dist = summary['source_type_distribution']
    print(f"\n🔗 Source Type Distribution:")
    for source_type, count in source_dist.items():
        print(f"   • {source_type}: {count} pairs")
    
    # Load processed data for sample analysis
    processed_file = Path("data/processed/qa_pairs_processed.json")
    if processed_file.exists():
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = data.get('qa_pairs', [])
        
        print(f"\n📝 Sample Processed Q&A Pairs:")
        for i, pair in enumerate(qa_pairs[:3]):
            print(f"\n{i+1}. {pair['question']}")
            print(f"   Answer: {pair['answer'][:150]}...")
            print(f"   Source: {pair['source']}")
            print(f"   Category: {pair['category']}")
            print(f"   Quality Score: {pair.get('processing_metadata', {}).get('quality_score', 0):.3f}")
            print(f"   Language: {pair.get('language_tag', 'unknown')}")

def check_duplicates_removed():
    """Check what types of duplicates were removed"""
    print(f"\n🔍 Duplicate Analysis:")
    print("=" * 60)
    
    # Load original data
    original_file = Path("data/raw/qa_pairs.json")
    processed_file = Path("data/processed/qa_pairs_processed.json")
    
    if not original_file.exists() or not processed_file.exists():
        print("❌ Cannot find original or processed data")
        return
    
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(processed_file, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    original_pairs = original_data.get('qa_pairs', [])
    processed_pairs = processed_data.get('qa_pairs', [])
    
    # Analyze what was removed
    original_by_type = {}
    processed_by_type = {}
    
    for pair in original_pairs:
        content_type = pair.get('content_type', 'unknown')
        original_by_type[content_type] = original_by_type.get(content_type, 0) + 1
    
    for pair in processed_pairs:
        content_type = pair.get('content_type', 'unknown')
        processed_by_type[content_type] = processed_by_type.get(content_type, 0) + 1
    
    print("📊 Duplicate Removal by Content Type:")
    for content_type in original_by_type:
        original_count = original_by_type[content_type]
        processed_count = processed_by_type.get(content_type, 0)
        removed_count = original_count - processed_count
        removal_rate = removed_count / original_count if original_count > 0 else 0
        
        print(f"   • {content_type}:")
        print(f"     - Original: {original_count}")
        print(f"     - Processed: {processed_count}")
        print(f"     - Removed: {removed_count} ({removal_rate:.1%})")

if __name__ == "__main__":
    analyze_processing_results()
    check_duplicates_removed() 