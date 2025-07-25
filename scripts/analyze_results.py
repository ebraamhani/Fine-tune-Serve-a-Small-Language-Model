#!/usr/bin/env python3
"""
Analyze Data Collection Results
"""

import json
from pathlib import Path

def analyze_qa_pairs():
    """Analyze the generated Q&A pairs"""
    qa_file = Path("data/raw/qa_pairs.json")
    
    if not qa_file.exists():
        print("❌ Q&A pairs file not found")
        return
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("📊 Q&A Pairs Analysis")
    print("=" * 50)
    print(f"Total Q&A pairs: {data['metadata']['total_pairs']}")
    print(f"Sources: {len(data['metadata']['sources'])}")
    
    # Analyze categories
    categories = {}
    content_types = {}
    
    for pair in data['qa_pairs']:
        category = pair['category']
        content_type = pair['content_type']
        
        categories[category] = categories.get(category, 0) + 1
        content_types[content_type] = content_types.get(content_type, 0) + 1
    
    print(f"\n📂 Categories ({len(categories)}):")
    for category, count in sorted(categories.items()):
        print(f"   • {category}: {count}")
    
    print(f"\n🌐 Content Types ({len(content_types)}):")
    for content_type, count in sorted(content_types.items()):
        print(f"   • {content_type}: {count}")
    
    # Show sample Q&A pairs
    print(f"\n📝 Sample Q&A Pairs:")
    for i, pair in enumerate(data['qa_pairs'][:5]):
        print(f"\n{i+1}. {pair['question']}")
        print(f"   Answer: {pair['answer'][:200]}...")
        print(f"   Source: {pair['source']}")
        print(f"   Category: {pair['category']}")

def analyze_collection_summary():
    """Analyze the collection summary"""
    summary_file = Path("data/raw/metadata/overall_collection_summary.json")
    
    if not summary_file.exists():
        print("❌ Collection summary not found")
        return
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n📋 Collection Summary")
    print("=" * 50)
    
    session = data['collection_session']
    print(f"Duration: {session['duration_minutes']:.1f} minutes")
    
    web_collection = data['web_collection']
    print(f"Web sources: {web_collection['statistics']['total_sources']}")
    print(f"Total words: {web_collection['statistics']['total_words']:,}")
    
    # Show source breakdown
    sources = web_collection['sources']
    print(f"\n📚 Source Breakdown:")
    for source_type, source_list in sources.items():
        print(f"   • {source_type}: {len(source_list)} sources")
        total_words = sum(s['word_count'] for s in source_list)
        print(f"     Total words: {total_words:,}")

if __name__ == "__main__":
    analyze_qa_pairs()
    analyze_collection_summary() 