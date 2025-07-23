"""
Web Scraper for Tourist QA Assistant
Collects domain-specific data about Egypt tourism from various sources
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
import re
from typing import Dict, List, Optional, Any
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EgyptTourismScraper:
    """Web scraper for Egypt tourism information"""
    
    def __init__(self, output_dir: str = "data/raw/web"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Headers to mimic browser requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Session for persistent connections
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Rate limiting
        self.delay = 2  # seconds between requests
        
    def scrape_wikipedia_egypt(self) -> List[Dict[str, Any]]:
        """Scrape Wikipedia pages about Egypt tourism"""
        logger.info("Scraping Wikipedia pages about Egypt tourism...")
        
        wiki_pages = [
            "https://en.wikipedia.org/wiki/Tourism_in_Egypt",
            "https://en.wikipedia.org/wiki/Ancient_Egypt",
            "https://en.wikipedia.org/wiki/Egyptian_pyramids",
            "https://en.wikipedia.org/wiki/Valley_of_the_Kings",
            "https://en.wikipedia.org/wiki/Egyptian_Museum",
            "https://en.wikipedia.org/wiki/Cairo",
            "https://en.wikipedia.org/wiki/Luxor",
            "https://en.wikipedia.org/wiki/Aswan",
            "https://en.wikipedia.org/wiki/Alexandria",
            "https://en.wikipedia.org/wiki/Sharm_El_Sheikh"
        ]
        
        scraped_data = []
        
        for url in wiki_pages:
            try:
                logger.info(f"Scraping: {url}")
                data = self._scrape_wikipedia_page(url)
                if data:
                    scraped_data.append(data)
                time.sleep(self.delay)
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                
        return scraped_data
    
    def _scrape_wikipedia_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single Wikipedia page"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('h1', {'id': 'firstHeading'})
            title_text = title.get_text().strip() if title else "Unknown Title"
            
            # Extract main content
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
                
            # Remove unwanted elements
            for element in content_div.find_all(['script', 'style', 'sup', 'table']):
                element.decompose()
                
            # Extract paragraphs
            paragraphs = []
            for p in content_div.find_all('p'):
                text = p.get_text().strip()
                if text and len(text) > 50:  # Filter out short paragraphs
                    paragraphs.append(text)
                    
            # Extract headings
            headings = []
            for heading in content_div.find_all(['h2', 'h3', 'h4']):
                text = heading.get_text().strip()
                if text and not text.startswith('['):  # Filter out edit links
                    headings.append({
                        'level': heading.name,
                        'text': text
                    })
            
            # Create metadata
            metadata = {
                'source_url': url,
                'title': title_text,
                'scraped_at': datetime.now().isoformat(),
                'content_type': 'wikipedia',
                'language': 'en',
                'word_count': sum(len(p.split()) for p in paragraphs),
                'paragraph_count': len(paragraphs),
                'heading_count': len(headings)
            }
            
            # Create data structure
            data = {
                'metadata': metadata,
                'content': {
                    'title': title_text,
                    'paragraphs': paragraphs,
                    'headings': headings
                }
            }
            
            # Save to file
            self._save_data(data, url)
            
            return data
            
        except Exception as e:
            logger.error(f"Error scraping Wikipedia page {url}: {e}")
            return None
    
    def scrape_government_sites(self) -> List[Dict[str, Any]]:
        """Scrape Egyptian government tourism websites"""
        logger.info("Scraping Egyptian government tourism websites...")
        
        gov_sites = [
            "https://www.egypt.travel/",
            "https://www.visa2egypt.gov.eg/",
            "https://www.egyptair.com/",
            "https://www.egypt.travel/visa-information",
            "https://www.egypt.travel/safety-tips"
        ]
        
        scraped_data = []
        
        for url in gov_sites:
            try:
                logger.info(f"Scraping: {url}")
                data = self._scrape_general_page(url)
                if data:
                    scraped_data.append(data)
                time.sleep(self.delay)
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                
        return scraped_data
    
    def _scrape_general_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a general web page"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Unknown Title"
            
            # Extract main content (try different selectors)
            content_selectors = [
                'main', 'article', '.content', '.main-content', 
                '#content', '#main', '.post-content', '.entry-content',
                '.article-content', '.page-content', 'body'
            ]
            
            content_element = None
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    break
            
            if not content_element:
                content_element = soup.find('body')
            
            # Remove unwanted elements more aggressively
            unwanted_selectors = [
                'script', 'style', 'nav', 'header', 'footer', 'aside',
                '.navigation', '.menu', '.sidebar', '.advertisement',
                '.social-share', '.comments', '.related-posts',
                '[class*="nav"]', '[class*="menu"]', '[class*="sidebar"]',
                '[class*="ad"]', '[class*="social"]', '[class*="comment"]'
            ]
            
            for selector in unwanted_selectors:
                for element in content_element.find_all(selector):
                    element.decompose()
            
            # Extract paragraphs and meaningful text
            paragraphs = []
            for p in content_element.find_all(['p', 'div', 'section']):
                text = p.get_text().strip()
                # Filter out short or navigation-like text
                if (len(text) > 50 and 
                    not text.startswith('Menu') and
                    not text.startswith('Navigation') and
                    not text.startswith('Search') and
                    not text.startswith('Login') and
                    not text.startswith('Sign up') and
                    not text.startswith('Follow us') and
                    not text.startswith('Share') and
                    not text.startswith('Cookie') and
                    not text.startswith('Privacy') and
                    not text.startswith('Terms') and
                    not text.startswith('©') and
                    not text.startswith('All rights reserved')):
                    paragraphs.append(text)
            
            # Join meaningful paragraphs
            cleaned_text = '\n\n'.join(paragraphs)
            
            # If we don't have enough meaningful content, try a different approach
            if len(cleaned_text.split()) < 100:
                # Try to extract text from specific content areas
                content_areas = content_element.find_all(['article', 'section', 'div'], 
                                                       class_=lambda x: x and any(word in x.lower() 
                                                                                  for word in ['content', 'article', 'post', 'entry', 'main']))
                if content_areas:
                    for area in content_areas:
                        text = area.get_text().strip()
                        if len(text) > 200:
                            cleaned_text = text
                            break
            
            # Create metadata
            metadata = {
                'source_url': url,
                'title': title_text,
                'scraped_at': datetime.now().isoformat(),
                'content_type': 'government_site',
                'language': 'en',
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text)
            }
            
            # Create data structure
            data = {
                'metadata': metadata,
                'content': {
                    'title': title_text,
                    'text': cleaned_text
                }
            }
            
            # Save to file
            self._save_data(data, url)
            
            return data
            
        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return None
    
    def scrape_travel_guides(self) -> List[Dict[str, Any]]:
        """Scrape travel guide websites"""
        logger.info("Scraping travel guide websites...")
        
        travel_sites = [
            "https://www.lonelyplanet.com/egypt",
            "https://www.roughguides.com/egypt/",
            "https://www.tripadvisor.com/Attractions-g294200-Activities-Egypt.html",
            "https://www.viator.com/Egypt-attractions/d824-a2207"
        ]
        
        scraped_data = []
        
        for url in travel_sites:
            try:
                logger.info(f"Scraping: {url}")
                data = self._scrape_general_page(url)
                if data:
                    data['metadata']['content_type'] = 'travel_guide'
                    scraped_data.append(data)
                time.sleep(self.delay)
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                
        return scraped_data
    
    def _save_data(self, data: Dict[str, Any], url: str) -> None:
        """Save scraped data to file with metadata"""
        try:
            # Create filename from URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace('.', '_')
            path = parsed_url.path.replace('/', '_').replace('.', '_')
            if not path:
                path = 'index'
            
            # Create unique filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"{domain}_{path}_{url_hash}.json"
            
            # Save data
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data for {url}: {e}")
    
    def create_metadata_summary(self) -> Dict[str, Any]:
        """Create a summary of all scraped data"""
        logger.info("Creating metadata summary...")
        
        summary = {
            'scraping_session': {
                'started_at': datetime.now().isoformat(),
                'total_files': 0,
                'total_words': 0,
                'sources': {
                    'wikipedia': 0,
                    'government_site': 0,
                    'travel_guide': 0
                }
            },
            'files': []
        }
        
        # Scan all JSON files in output directory
        for json_file in self.output_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                summary['scraping_session']['total_files'] += 1
                summary['scraping_session']['total_words'] += metadata.get('word_count', 0)
                
                content_type = metadata.get('content_type', 'unknown')
                if content_type in summary['scraping_session']['sources']:
                    summary['scraping_session']['sources'][content_type] += 1
                
                summary['files'].append({
                    'filename': json_file.name,
                    'source_url': metadata.get('source_url'),
                    'title': metadata.get('title'),
                    'content_type': content_type,
                    'word_count': metadata.get('word_count', 0),
                    'scraped_at': metadata.get('scraped_at')
                })
                
            except Exception as e:
                logger.error(f"Error reading {json_file}: {e}")
        
        # Save summary
        summary_file = self.output_dir / "scraping_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Summary saved: {summary_file}")
        return summary
    
    def run_full_scraping(self) -> Dict[str, Any]:
        """Run complete scraping pipeline"""
        logger.info("Starting full scraping pipeline...")
        
        # Scrape different sources
        wiki_data = self.scrape_wikipedia_egypt()
        gov_data = self.scrape_government_sites()
        travel_data = self.scrape_travel_guides()
        
        # Create summary
        summary = self.create_metadata_summary()
        
        logger.info(f"Scraping completed! Total files: {summary['scraping_session']['total_files']}")
        
        return summary

if __name__ == "__main__":
    # Run the scraper
    scraper = EgyptTourismScraper()
    summary = scraper.run_full_scraping()
    print(json.dumps(summary, indent=2)) 