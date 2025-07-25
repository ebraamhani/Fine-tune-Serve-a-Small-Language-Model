#!/usr/bin/env python3
"""
Validate Processed Data Structure and Quality
"""

import json
from pathlib import Path
from typing import Dict, List, Any

def validate_processed_data():
    """Validate the processed data structure and quality"""
    
    print("🔍 Validating Processed Data")
    print("=" * 50)
    
    # Check main processed file
    processed_file = Path("data/processed/qa_pairs_processed.json")
    if not processed_file.exists():
        print("❌ Processed data file not found")
        return False
    
    with open(processed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate structure
    required_keys = ['metadata', 'qa_pairs']
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing required key: {key}")
            return False
    
    qa_pairs = data['qa_pairs']
    metadata = data['metadata']
    
    print(f"✅ Found {len(qa_pairs)} processed Q&A pairs")
    
    # Validate each Q&A pair structure
    required_pair_keys = [
        'question', 'answer', 'source', 'content_type', 
        'category', 'language_tag', 'language_info', 
        'processing_metadata', 'source_type'
    ]
    
    validation_errors = []
    
    for i, pair in enumerate(qa_pairs):
        # Check required keys
        for key in required_pair_keys:
            if key not in pair:
                validation_errors.append(f"Pair {i}: Missing key '{key}'")
        
        # Check data types and content
        if not isinstance(pair.get('question', ''), str) or not pair.get('question', '').strip():
            validation_errors.append(f"Pair {i}: Invalid or empty question")
        
        if not isinstance(pair.get('answer', ''), str) or not pair.get('answer', '').strip():
            validation_errors.append(f"Pair {i}: Invalid or empty answer")
        
        # Check language info structure
        lang_info = pair.get('language_info', {})
        if not isinstance(lang_info, dict):
            validation_errors.append(f"Pair {i}: Invalid language_info structure")
        else:
            required_lang_keys = ['question_language', 'answer_language', 'primary_language']
            for key in required_lang_keys:
                if key not in lang_info:
                    validation_errors.append(f"Pair {i}: Missing language_info key '{key}'")
        
        # Check processing metadata structure
        proc_meta = pair.get('processing_metadata', {})
        if not isinstance(proc_meta, dict):
            validation_errors.append(f"Pair {i}: Invalid processing_metadata structure")
        else:
            required_meta_keys = ['processed_at', 'question_length', 'answer_length', 'word_count', 'quality_score']
            for key in required_meta_keys:
                if key not in proc_meta:
                    validation_errors.append(f"Pair {i}: Missing processing_metadata key '{key}'")
    
    if validation_errors:
        print("❌ Validation errors found:")
        for error in validation_errors[:10]:  # Show first 10 errors
            print(f"   • {error}")
        if len(validation_errors) > 10:
            print(f"   ... and {len(validation_errors) - 10} more errors")
        return False
    
    print("✅ All Q&A pairs have valid structure")
    
    # Check quality metrics
    quality_scores = [p.get('processing_metadata', {}).get('quality_score', 0) for p in qa_pairs]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    print(f"📊 Quality Metrics:")
    print(f"   • Average quality score: {avg_quality:.3f}")
    print(f"   • Perfect quality pairs: {sum(1 for s in quality_scores if s == 1.0)}")
    print(f"   • High quality pairs (≥0.8): {sum(1 for s in quality_scores if s >= 0.8)}")
    
    # Check language distribution
    lang_counts = {}
    for pair in qa_pairs:
        lang = pair.get('language_tag', 'unknown')
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    print(f"🌐 Language Distribution:")
    for lang, count in lang_counts.items():
        print(f"   • {lang.upper()}: {count} pairs")
    
    # Check content type distribution
    content_counts = {}
    for pair in qa_pairs:
        content_type = pair.get('content_type', 'unknown')
        content_counts[content_type] = content_counts.get(content_type, 0) + 1
    
    print(f"📚 Content Type Distribution:")
    for content_type, count in content_counts.items():
        print(f"   • {content_type}: {count} pairs")
    
    # Check source type distribution
    source_counts = {}
    for pair in qa_pairs:
        source_type = pair.get('source_type', 'unknown')
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
    
    print(f"🔗 Source Type Distribution:")
    for source_type, count in source_counts.items():
        print(f"   • {source_type}: {count} pairs")
    
    # Check for duplicates (should be none)
    print(f"\n🔍 Duplicate Check:")
    seen_questions = set()
    duplicates = 0
    
    for pair in qa_pairs:
        question = pair.get('question', '').lower().strip()
        if question in seen_questions:
            duplicates += 1
        else:
            seen_questions.add(question)
    
    if duplicates == 0:
        print("✅ No duplicate questions found")
    else:
        print(f"⚠️  Found {duplicates} duplicate questions")
    
    # Check English-only file
    en_file = Path("data/processed/qa_pairs_en.json")
    if en_file.exists():
        with open(en_file, 'r', encoding='utf-8') as f:
            en_data = json.load(f)
        
        en_pairs = en_data.get('qa_pairs', [])
        print(f"✅ English-only file: {len(en_pairs)} pairs")
        
        # Verify all are English
        non_english = [p for p in en_pairs if p.get('language_tag') != 'en']
        if non_english:
            print(f"⚠️  Found {len(non_english)} non-English pairs in English file")
        else:
            print("✅ All pairs in English file are properly tagged")
    
    # Check summary file
    summary_file = Path("data/processed/metadata/processing_summary.json")
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print(f"✅ Processing summary: {summary['processing_session']['processed_count']} pairs")
    
    print("\n🎉 Data validation completed successfully!")
    return True

def check_data_quality_samples():
    """Check sample data quality"""
    
    print(f"\n📝 Sample Quality Check")
    print("=" * 50)
    
    processed_file = Path("data/processed/qa_pairs_processed.json")
    if not processed_file.exists():
        return
    
    with open(processed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = data['qa_pairs']
    
    # Show samples from different categories
    categories = {}
    for pair in qa_pairs:
        category = pair.get('category', 'unknown')
        if category not in categories:
            categories[category] = []
        categories[category].append(pair)
    
    print("📂 Sample Q&A pairs by category:")
    
    for category, pairs in list(categories.items())[:5]:  # Show first 5 categories
        if pairs:
            sample = pairs[0]
            print(f"\n• {category.upper()}:")
            print(f"  Q: {sample['question']}")
            print(f"  A: {sample['answer'][:100]}...")
            print(f"  Quality: {sample.get('processing_metadata', {}).get('quality_score', 0):.3f}")
            print(f"  Source: {sample['source']}")

if __name__ == "__main__":
    success = validate_processed_data()
    if success:
        check_data_quality_samples() 