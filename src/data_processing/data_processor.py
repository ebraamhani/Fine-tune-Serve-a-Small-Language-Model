"""
Data Processor for Tourist QA Assistant
Handles deduplication, quality checks, and data cleaning
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple
import hashlib
import re
from difflib import SequenceMatcher
import langdetect
from langdetect import detect, DetectorFactory
import pandas as pd

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TourismDataProcessor:
    """Data processor for tourism Q&A pairs"""
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "cleaned").mkdir(exist_ok=True)
        (self.output_dir / "deduplicated").mkdir(exist_ok=True)
        (self.output_dir / "quality_checked").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        # Quality thresholds
        self.min_answer_length = 20
        self.max_answer_length = 2000
        self.min_question_length = 10
        self.max_question_length = 200
        self.similarity_threshold = 0.85
        
    def load_qa_pairs(self) -> List[Dict[str, Any]]:
        """Load Q&A pairs from raw data"""
        qa_file = self.input_dir / "qa_pairs.json"
        
        if not qa_file.exists():
            raise FileNotFoundError(f"Q&A pairs file not found: {qa_file}")
        
        logger.info(f"Loading Q&A pairs from {qa_file}")
        
        with open(qa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = data.get('qa_pairs', [])
        logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")
        
        return qa_pairs
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            if not text or len(text.strip()) < 10:
                return "unknown"
            
            # Clean text for better detection
            clean_text = re.sub(r'[^\w\s]', '', text)
            if len(clean_text.strip()) < 5:
                return "unknown"
            
            lang = detect(clean_text)
            return lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown"
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity
    
    def is_duplicate(self, pair1: Dict[str, Any], pair2: Dict[str, Any]) -> bool:
        """Check if two Q&A pairs are duplicates"""
        # Check question similarity
        question_similarity = self.calculate_similarity(
            pair1['question'], pair2['question']
        )
        
        # Check answer similarity
        answer_similarity = self.calculate_similarity(
            pair1['answer'], pair2['answer']
        )
        
        # Consider duplicates if both question and answer are very similar
        if question_similarity > self.similarity_threshold and answer_similarity > self.similarity_threshold:
            return True
        
        # Also check if questions are very similar (even if answers differ)
        if question_similarity > 0.95:
            return True
        
        return False
    
    def quality_check(self, qa_pair: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Perform quality checks on a Q&A pair"""
        issues = []
        
        # Check question length
        question = qa_pair.get('question', '')
        if len(question) < self.min_question_length:
            issues.append(f"Question too short: {len(question)} chars")
        elif len(question) > self.max_question_length:
            issues.append(f"Question too long: {len(question)} chars")
        
        # Check answer length
        answer = qa_pair.get('answer', '')
        if len(answer) < self.min_answer_length:
            issues.append(f"Answer too short: {len(answer)} chars")
        elif len(answer) > self.max_answer_length:
            issues.append(f"Answer too long: {len(answer)} chars")
        
        # Check for empty or invalid content
        if not question.strip():
            issues.append("Empty question")
        if not answer.strip():
            issues.append("Empty answer")
        
        # Check for repetitive content
        if question.lower() == answer.lower():
            issues.append("Question and answer are identical")
        
        # Check for generic answers
        generic_phrases = [
            "this document contains",
            "this information is provided",
            "please refer to",
            "contact us for",
            "for more information"
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in generic_phrases):
            issues.append("Generic answer detected")
        
        # Check for source validity
        source = qa_pair.get('source', '')
        if not source:
            issues.append("Missing source")
        
        return len(issues) == 0, issues
    
    def deduplicate_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate Q&A pairs"""
        logger.info("Starting deduplication...")
        
        # Create hash for each pair for faster comparison
        def create_hash(pair: Dict[str, Any]) -> str:
            content = f"{pair.get('question', '')} {pair.get('answer', '')}"
            return hashlib.md5(content.encode()).hexdigest()
        
        # Group by content type for more efficient deduplication
        content_groups = {}
        for pair in qa_pairs:
            content_type = pair.get('content_type', 'unknown')
            if content_type not in content_groups:
                content_groups[content_type] = []
            content_groups[content_type].append(pair)
        
        deduplicated_pairs = []
        duplicates_found = 0
        
        # Process each content type separately
        for content_type, pairs in content_groups.items():
            logger.info(f"Deduplicating {len(pairs)} pairs from {content_type}")
            
            content_deduplicated = []
            seen_hashes = set()
            
            for pair in pairs:
                pair_hash = create_hash(pair)
                
                # Check if we've seen this exact content before
                if pair_hash in seen_hashes:
                    duplicates_found += 1
                    continue
                
                # Check for similar pairs
                is_duplicate = False
                for existing_pair in content_deduplicated:
                    if self.is_duplicate(pair, existing_pair):
                        duplicates_found += 1
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    content_deduplicated.append(pair)
                    seen_hashes.add(pair_hash)
            
            deduplicated_pairs.extend(content_deduplicated)
            logger.info(f"  Kept {len(content_deduplicated)} pairs from {content_type}")
        
        logger.info(f"Deduplication completed: {duplicates_found} duplicates removed")
        return deduplicated_pairs
    
    def add_language_tags(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add language tags to Q&A pairs"""
        logger.info("Adding language tags...")
        
        for pair in qa_pairs:
            # Detect language for question and answer
            question_lang = self.detect_language(pair.get('question', ''))
            answer_lang = self.detect_language(pair.get('answer', ''))
            
            # Add language information
            pair['language_info'] = {
                'question_language': question_lang,
                'answer_language': answer_lang,
                'primary_language': answer_lang if answer_lang != 'unknown' else question_lang
            }
            
            # Add language tag for future filtering
            primary_lang = pair['language_info']['primary_language']
            if primary_lang in ['en', 'ar']:
                pair['language_tag'] = primary_lang
            else:
                pair['language_tag'] = 'unknown'
        
        # Count languages
        lang_counts = {}
        for pair in qa_pairs:
            lang = pair.get('language_tag', 'unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        logger.info(f"Language distribution: {lang_counts}")
        return qa_pairs
    
    def enhance_metadata(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance Q&A pairs with additional metadata"""
        logger.info("Enhancing metadata...")
        
        for pair in qa_pairs:
            # Add processing metadata
            pair['processing_metadata'] = {
                'processed_at': datetime.now().isoformat(),
                'question_length': len(pair.get('question', '')),
                'answer_length': len(pair.get('answer', '')),
                'word_count': len(pair.get('answer', '').split()),
                'quality_score': self._calculate_quality_score(pair)
            }
            
            # Add source type classification
            source = pair.get('source', '')
            if 'wikipedia' in source.lower():
                pair['source_type'] = 'encyclopedia'
            elif 'gov' in source.lower() or 'egypt.travel' in source.lower():
                pair['source_type'] = 'official'
            elif 'pdf' in source.lower():
                pair['source_type'] = 'document'
            else:
                pair['source_type'] = 'travel_guide'
        
        return qa_pairs
    
    def _calculate_quality_score(self, pair: Dict[str, Any]) -> float:
        """Calculate a quality score for a Q&A pair"""
        score = 1.0
        
        # Length penalties
        question_len = len(pair.get('question', ''))
        answer_len = len(pair.get('answer', ''))
        
        if question_len < 20 or question_len > 150:
            score -= 0.2
        if answer_len < 50 or answer_len > 1000:
            score -= 0.2
        
        # Source quality bonus
        source = pair.get('source', '')
        if 'wikipedia' in source.lower():
            score += 0.1
        elif 'gov' in source.lower():
            score += 0.1
        
        # Content type bonus
        content_type = pair.get('content_type', '')
        if content_type == 'wikipedia':
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def create_processing_summary(self, original_pairs: List[Dict[str, Any]], 
                                processed_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of the processing results"""
        summary = {
            'processing_session': {
                'started_at': datetime.now().isoformat(),
                'original_count': len(original_pairs),
                'processed_count': len(processed_pairs),
                'removed_count': len(original_pairs) - len(processed_pairs),
                'retention_rate': len(processed_pairs) / len(original_pairs) if original_pairs else 0
            },
            'quality_metrics': {
                'avg_question_length': sum(len(p.get('question', '')) for p in processed_pairs) / len(processed_pairs) if processed_pairs else 0,
                'avg_answer_length': sum(len(p.get('answer', '')) for p in processed_pairs) / len(processed_pairs) if processed_pairs else 0,
                'avg_quality_score': sum(p.get('processing_metadata', {}).get('quality_score', 0) for p in processed_pairs) / len(processed_pairs) if processed_pairs else 0
            },
            'language_distribution': {},
            'content_type_distribution': {},
            'category_distribution': {},
            'source_type_distribution': {}
        }
        
        # Calculate distributions
        for pair in processed_pairs:
            # Language
            lang = pair.get('language_tag', 'unknown')
            summary['language_distribution'][lang] = summary['language_distribution'].get(lang, 0) + 1
            
            # Content type
            content_type = pair.get('content_type', 'unknown')
            summary['content_type_distribution'][content_type] = summary['content_type_distribution'].get(content_type, 0) + 1
            
            # Category
            category = pair.get('category', 'unknown')
            summary['category_distribution'][category] = summary['category_distribution'].get(category, 0) + 1
            
            # Source type
            source_type = pair.get('source_type', 'unknown')
            summary['source_type_distribution'][source_type] = summary['source_type_distribution'].get(source_type, 0) + 1
        
        return summary
    
    def save_processed_data(self, qa_pairs: List[Dict[str, Any]], 
                          summary: Dict[str, Any]) -> None:
        """Save processed data and summary"""
        # Save processed Q&A pairs
        processed_file = self.output_dir / "qa_pairs_processed.json"
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_pairs': len(qa_pairs),
                    'processing_summary': summary
                },
                'qa_pairs': qa_pairs
            }, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = self.output_dir / "metadata" / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save language-specific files
        lang_groups = {}
        for pair in qa_pairs:
            lang = pair.get('language_tag', 'unknown')
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append(pair)
        
        for lang, pairs in lang_groups.items():
            if lang != 'unknown':
                lang_file = self.output_dir / f"qa_pairs_{lang}.json"
                with open(lang_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'metadata': {
                            'language': lang,
                            'total_pairs': len(pairs),
                            'created_at': datetime.now().isoformat()
                        },
                        'qa_pairs': pairs
                    }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed data saved to {self.output_dir}")
    
    def run_full_processing(self) -> Dict[str, Any]:
        """Run complete data processing pipeline"""
        logger.info("Starting full data processing pipeline...")
        
        # Load original data
        original_pairs = self.load_qa_pairs()
        
        # Clean text
        logger.info("Cleaning text...")
        for pair in original_pairs:
            pair['question'] = self.clean_text(pair.get('question', ''))
            pair['answer'] = self.clean_text(pair.get('answer', ''))
        
        # Quality check
        logger.info("Performing quality checks...")
        quality_issues = []
        quality_passed = []
        
        for pair in original_pairs:
            passed, issues = self.quality_check(pair)
            if passed:
                quality_passed.append(pair)
            else:
                quality_issues.append({
                    'pair': pair,
                    'issues': issues
                })
        
        logger.info(f"Quality check: {len(quality_passed)} passed, {len(quality_issues)} failed")
        
        # Deduplicate
        deduplicated_pairs = self.deduplicate_qa_pairs(quality_passed)
        
        # Add language tags
        lang_tagged_pairs = self.add_language_tags(deduplicated_pairs)
        
        # Enhance metadata
        final_pairs = self.enhance_metadata(lang_tagged_pairs)
        
        # Create summary
        summary = self.create_processing_summary(original_pairs, final_pairs)
        
        # Save processed data
        self.save_processed_data(final_pairs, summary)
        
        logger.info("Data processing completed!")
        return summary

if __name__ == "__main__":
    processor = TourismDataProcessor()
    summary = processor.run_full_processing()
    print(json.dumps(summary, indent=2)) 