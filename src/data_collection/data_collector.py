"""
Main Data Collector for Tourist QA Assistant
Orchestrates web scraping and PDF extraction with comprehensive metadata management
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
import time

from .web_scraper import EgyptTourismScraper
from .pdf_extractor import TourismPDFExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TourismDataCollector:
    """Main data collector for Egypt tourism information"""
    
    def __init__(self, base_output_dir: str = "data/raw"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-collectors
        self.web_scraper = EgyptTourismScraper(output_dir=str(self.base_output_dir / "web"))
        self.pdf_extractor = TourismPDFExtractor(output_dir=str(self.base_output_dir / "pdf"))
        
        # Create metadata directory
        self.metadata_dir = self.base_output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
    def collect_web_data(self, include_travel_guides: bool = True) -> Dict[str, Any]:
        """Collect data from web sources"""
        logger.info("Starting web data collection...")
        
        collection_summary = {
            'collection_type': 'web',
            'started_at': datetime.now().isoformat(),
            'sources': {
                'wikipedia': [],
                'government_sites': [],
                'travel_guides': []
            },
            'statistics': {
                'total_sources': 0,
                'total_words': 0,
                'total_files': 0
            }
        }
        
        # Scrape Wikipedia
        logger.info("Scraping Wikipedia pages...")
        wiki_data = self.web_scraper.scrape_wikipedia_egypt()
        collection_summary['sources']['wikipedia'] = [
            {
                'title': item['metadata']['title'],
                'url': item['metadata']['source_url'],
                'word_count': item['metadata']['word_count']
            } for item in wiki_data
        ]
        
        # Scrape government sites
        logger.info("Scraping government sites...")
        gov_data = self.web_scraper.scrape_government_sites()
        collection_summary['sources']['government_sites'] = [
            {
                'title': item['metadata']['title'],
                'url': item['metadata']['source_url'],
                'word_count': item['metadata']['word_count']
            } for item in gov_data
        ]
        
        # Scrape travel guides (optional)
        if include_travel_guides:
            logger.info("Scraping travel guides...")
            travel_data = self.web_scraper.scrape_travel_guides()
            collection_summary['sources']['travel_guides'] = [
                {
                    'title': item['metadata']['title'],
                    'url': item['metadata']['source_url'],
                    'word_count': item['metadata']['word_count']
                } for item in travel_data
            ]
        
        # Calculate statistics
        all_data = wiki_data + gov_data + (travel_data if include_travel_guides else [])
        collection_summary['statistics']['total_sources'] = len(all_data)
        collection_summary['statistics']['total_words'] = sum(
            item['metadata']['word_count'] for item in all_data
        )
        collection_summary['statistics']['total_files'] = len(all_data)
        
        # Save collection summary
        self._save_collection_summary(collection_summary, 'web_collection')
        
        logger.info(f"Web data collection completed! Total sources: {collection_summary['statistics']['total_sources']}")
        return collection_summary
    
    def collect_pdf_data(self, pdf_dir: Optional[str] = None, extract_images: bool = False) -> Dict[str, Any]:
        """Collect data from PDF sources"""
        logger.info("Starting PDF data collection...")
        
        if pdf_dir is None:
            pdf_dir = self.base_output_dir / "pdf_samples"
            pdf_dir.mkdir(exist_ok=True)
            logger.info(f"No PDF directory specified, using: {pdf_dir}")
        
        collection_summary = {
            'collection_type': 'pdf',
            'started_at': datetime.now().isoformat(),
            'pdf_directory': str(pdf_dir),
            'extract_images': extract_images,
            'statistics': {
                'total_files': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'total_words': 0,
                'total_tables': 0
            }
        }
        
        # Process PDFs
        pdf_summary = self.pdf_extractor.process_pdf_directory(pdf_dir, extract_images)
        
        # Update collection summary with PDF results
        collection_summary['statistics'] = pdf_summary['processing_session']
        collection_summary['files'] = pdf_summary['files']
        
        # Save collection summary
        self._save_collection_summary(collection_summary, 'pdf_collection')
        
        logger.info(f"PDF data collection completed! Total files: {collection_summary['statistics']['total_files']}")
        return collection_summary
    
    def create_qa_pairs(self) -> List[Dict[str, Any]]:
        """Create Q&A pairs from collected data"""
        logger.info("Creating Q&A pairs from collected data...")
        
        qa_pairs = []
        
        # Load web data
        web_dir = self.base_output_dir / "web"
        for json_file in web_dir.glob("*.json"):
            if json_file.name == "scraping_summary.json":
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Generate Q&A pairs based on content type
                content_type = data['metadata']['content_type']
                
                if content_type == 'wikipedia':
                    pairs = self._create_qa_from_wikipedia(data)
                elif content_type == 'government_site':
                    pairs = self._create_qa_from_government_site(data)
                elif content_type == 'travel_guide':
                    pairs = self._create_qa_from_travel_guide(data)
                else:
                    pairs = self._create_qa_from_general_content(data)
                
                qa_pairs.extend(pairs)
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        # Load PDF data
        pdf_dir = self.base_output_dir / "pdf"
        for json_file in pdf_dir.glob("*_extraction.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                pairs = self._create_qa_from_pdf(data)
                qa_pairs.extend(pairs)
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        # Save Q&A pairs
        qa_file = self.base_output_dir / "qa_pairs.json"
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_pairs': len(qa_pairs),
                    'sources': list(set(pair['source'] for pair in qa_pairs))
                },
                'qa_pairs': qa_pairs
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def _create_qa_from_wikipedia(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create Q&A pairs from Wikipedia content"""
        pairs = []
        content = data['content']
        metadata = data['metadata']
        
        # Create questions from headings
        for heading in content.get('headings', []):
            if heading['level'] == 'h2':  # Main sections
                question = f"What is {heading['text']} in Egypt?"
                answer = self._find_answer_for_heading(heading['text'], content['paragraphs'])
                
                if answer:
                    pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': metadata['source_url'],
                        'content_type': 'wikipedia',
                        'category': 'general_info'
                    })
        
        # Create specific tourism questions
        tourism_questions = [
            ("What are the main tourist attractions in Egypt?", "attractions"),
            ("What is the best time to visit Egypt?", "timing"),
            ("What should I know about Egyptian culture?", "culture"),
            ("How safe is it to travel in Egypt?", "safety"),
            ("What is the currency in Egypt?", "currency")
        ]
        
        for question, category in tourism_questions:
            answer = self._find_answer_in_content(question, content['paragraphs'])
            if answer:
                pairs.append({
                    'question': question,
                    'answer': answer,
                    'source': metadata['source_url'],
                    'content_type': 'wikipedia',
                    'category': category
                })
        
        return pairs
    
    def _create_qa_from_government_site(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create Q&A pairs from government site content"""
        pairs = []
        content = data['content']
        metadata = data['metadata']
        
        # Only create Q&A pairs if we have meaningful content
        if len(content['text'].split()) < 100:
            return pairs
        
        # Government sites typically contain practical information
        practical_questions = [
            ("Do I need a visa to visit Egypt?", "visa"),
            ("What documents do I need for Egypt visa?", "documents"),
            ("How much does an Egypt visa cost?", "costs"),
            ("What are the entry requirements for Egypt?", "entry_requirements"),
            ("What are the customs regulations in Egypt?", "customs"),
            ("What is the emergency number in Egypt?", "emergency"),
            ("What are the business hours in Egypt?", "business_hours"),
            ("What should I know about safety in Egypt?", "safety"),
            ("What are the health requirements for visiting Egypt?", "health"),
            ("What currency is used in Egypt?", "currency")
        ]
        
        for question, category in practical_questions:
            answer = self._find_answer_in_content(question, content['text'])
            if answer and len(answer.split()) > 20:  # Only use substantial answers
                pairs.append({
                    'question': question,
                    'answer': answer,
                    'source': metadata['source_url'],
                    'content_type': 'government_site',
                    'category': category
                })
        
        return pairs
    
    def _create_qa_from_travel_guide(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create Q&A pairs from travel guide content"""
        pairs = []
        content = data['content']
        metadata = data['metadata']
        
        # Travel guides contain experiential information
        travel_questions = [
            ("What should I pack for a trip to Egypt?", "packing"),
            ("What is the best way to get around Egypt?", "transportation"),
            ("What are the best hotels in Egypt?", "accommodation"),
            ("What should I eat in Egypt?", "food"),
            ("What souvenirs should I buy in Egypt?", "shopping"),
            ("What are the best tours in Egypt?", "tours"),
            ("What should I wear in Egypt?", "clothing"),
            ("What are the tipping customs in Egypt?", "tipping")
        ]
        
        for question, category in travel_questions:
            answer = self._find_answer_in_content(question, content['text'])
            if answer:
                pairs.append({
                    'question': question,
                    'answer': answer,
                    'source': metadata['source_url'],
                    'content_type': 'travel_guide',
                    'category': category
                })
        
        return pairs
    
    def _create_qa_from_pdf(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create Q&A pairs from PDF content"""
        pairs = []
        metadata = data['metadata']
        
        if 'text_extraction' in data and 'full_text' in data['text_extraction']:
            text = data['text_extraction']['full_text']
            
            # Create questions based on PDF content
            pdf_questions = [
                ("What information is provided in this document?", "document_info"),
                ("What are the key points in this document?", "key_points"),
                ("What procedures are described in this document?", "procedures"),
                ("What requirements are mentioned in this document?", "requirements")
            ]
            
            for question, category in pdf_questions:
                answer = self._find_answer_in_content(question, text)
                if answer:
                    pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': metadata['file_path'],
                        'content_type': 'pdf',
                        'category': category
                    })
        
        return pairs
    
    def _create_qa_from_general_content(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create Q&A pairs from general content"""
        pairs = []
        content = data['content']
        metadata = data['metadata']
        
        # Generic questions for any content
        general_questions = [
            ("What is the main topic of this content?", "topic"),
            ("What are the important details mentioned?", "details"),
            ("What should visitors know about this?", "visitor_info")
        ]
        
        text = content.get('text', '') or ' '.join(content.get('paragraphs', []))
        
        for question, category in general_questions:
            answer = self._find_answer_in_content(question, text)
            if answer:
                pairs.append({
                    'question': question,
                    'answer': answer,
                    'source': metadata['source_url'],
                    'content_type': metadata['content_type'],
                    'category': category
                })
        
        return pairs
    
    def _find_answer_for_heading(self, heading: str, paragraphs: List[str]) -> Optional[str]:
        """Find relevant answer for a specific heading"""
        # Simple keyword matching
        heading_keywords = heading.lower().split()
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            if any(keyword in paragraph_lower for keyword in heading_keywords):
                return paragraph[:500] + "..." if len(paragraph) > 500 else paragraph
        
        return None
    
    def _find_answer_in_content(self, question: str, content) -> Optional[str]:
        """Find relevant answer in content based on question"""
        # Handle different content types
        if isinstance(content, list):
            # For Wikipedia paragraphs
            text_content = ' '.join(content)
        elif isinstance(content, str):
            text_content = content
        else:
            return None
        
        # Simple keyword extraction from question
        question_lower = question.lower()
        keywords = [word for word in question_lower.split() if len(word) > 3]
        
        # Split content into sentences
        sentences = text_content.split('. ')
        
        # Find sentences with relevant keywords
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Return first few relevant sentences
            answer = '. '.join(relevant_sentences[:3])
            return answer[:500] + "..." if len(answer) > 500 else answer
        
        return None
    
    def _save_collection_summary(self, summary: Dict[str, Any], collection_type: str) -> None:
        """Save collection summary to metadata directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{collection_type}_{timestamp}.json"
        filepath = self.metadata_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved collection summary: {filepath}")
    
    def run_full_collection(self, include_travel_guides: bool = True, extract_images: bool = False) -> Dict[str, Any]:
        """Run complete data collection pipeline"""
        logger.info("Starting full data collection pipeline...")
        
        start_time = datetime.now()
        
        # Collect web data
        web_summary = self.collect_web_data(include_travel_guides)
        
        # Collect PDF data
        pdf_summary = self.collect_pdf_data(extract_images=extract_images)
        
        # Create Q&A pairs
        qa_pairs = self.create_qa_pairs()
        
        # Create overall summary
        overall_summary = {
            'collection_session': {
                'started_at': start_time.isoformat(),
                'completed_at': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
            },
            'web_collection': web_summary,
            'pdf_collection': pdf_summary,
            'qa_generation': {
                'total_pairs': len(qa_pairs),
                'categories': list(set(pair['category'] for pair in qa_pairs)),
                'content_types': list(set(pair['content_type'] for pair in qa_pairs))
            },
            'total_statistics': {
                'total_sources': web_summary['statistics']['total_sources'],
                'total_pdf_files': pdf_summary['statistics']['total_files'],
                'total_words': web_summary['statistics']['total_words'] + pdf_summary['statistics']['total_words'],
                'total_qa_pairs': len(qa_pairs)
            }
        }
        
        # Save overall summary
        overall_file = self.metadata_dir / "overall_collection_summary.json"
        with open(overall_file, 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, indent=2, ensure_ascii=False)
        
        logger.info("Full data collection pipeline completed!")
        logger.info(f"Total Q&A pairs created: {len(qa_pairs)}")
        logger.info(f"Overall summary saved: {overall_file}")
        
        return overall_summary

if __name__ == "__main__":
    # Run the complete data collection
    collector = TourismDataCollector()
    summary = collector.run_full_collection()
    print(json.dumps(summary, indent=2)) 