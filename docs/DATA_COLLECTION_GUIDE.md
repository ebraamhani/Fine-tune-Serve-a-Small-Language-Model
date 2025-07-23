# Data Collection Guide for Tourist QA Assistant

This guide explains how to use the data collection system for gathering Egypt tourism information from web sources and PDF documents.

## Overview

The data collection system consists of three main components:

1. **Web Scraper** - Collects data from Wikipedia, government sites, and travel guides
2. **PDF Extractor** - Extracts text and layout from PDF documents
3. **Data Collector** - Orchestrates the entire collection process and generates Q&A pairs

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Data Collection

```bash
python scripts/run_data_collection.py
```

This will:
- Scrape Wikipedia pages about Egypt tourism
- Collect data from government websites
- Process any PDF files in `data/raw/pdf_samples/`
- Generate Q&A pairs from all collected data

## Detailed Usage

### Web Scraping

The web scraper collects data from three types of sources:

#### Wikipedia Pages
- Tourism in Egypt
- Ancient Egypt
- Egyptian pyramids
- Valley of the Kings
- Egyptian Museum
- Major cities (Cairo, Luxor, Aswan, Alexandria, Sharm El Sheikh)

#### Government Sites
- egypt.travel (official tourism site)
- visa2egypt.gov.eg (visa information)
- egyptair.com (airline information)

#### Travel Guides
- Lonely Planet Egypt
- Rough Guides Egypt
- TripAdvisor Egypt attractions
- Viator Egypt tours

#### Running Web Scraping Only

```bash
python scripts/run_data_collection.py --web-only
```

### PDF Processing

The PDF extractor supports multiple extraction methods:

#### Text Extraction (PyMuPDF)
- Extracts plain text from PDFs
- Preserves page structure
- Handles most PDF formats

#### Layout Extraction (pdfplumber)
- Extracts text with positioning information
- Identifies tables and images
- Preserves formatting details

#### Image Extraction (pdf2image + OCR)
- Converts PDF pages to images
- Performs OCR on images
- Useful for scanned documents

#### Running PDF Processing Only

```bash
# Process PDFs in default directory
python scripts/run_data_collection.py --pdf-only

# Process PDFs in specific directory
python scripts/run_data_collection.py --pdf-only --pdf-dir /path/to/pdfs

# Include image extraction and OCR
python scripts/run_data_collection.py --pdf-only --extract-images
```

### Q&A Generation

The system automatically generates Q&A pairs from collected data:

#### Wikipedia Content
- Questions based on section headings
- Tourism-specific questions (attractions, timing, culture, safety, currency)

#### Government Site Content
- Practical questions (visa, documents, costs, entry requirements, customs, emergency)

#### Travel Guide Content
- Experiential questions (packing, transportation, accommodation, food, shopping, tours, clothing, tipping)

#### PDF Content
- Document-specific questions (information, key points, procedures, requirements)

#### Running Q&A Generation Only

```bash
python scripts/run_data_collection.py --qa-only
```

## Output Structure

```
data/raw/
├── web/                          # Web scraped data
│   ├── *.json                   # Individual scraped pages
│   └── scraping_summary.json    # Web scraping summary
├── pdf/                         # PDF extracted data
│   ├── text/                    # Plain text files
│   ├── layout/                  # Layout information
│   ├── images/                  # Extracted images
│   ├── *_extraction.json        # Full extraction data
│   └── pdf_extraction_summary.json
├── metadata/                    # Collection metadata
│   ├── web_collection_*.json    # Web collection summaries
│   ├── pdf_collection_*.json    # PDF collection summaries
│   └── overall_collection_summary.json
└── qa_pairs.json               # Generated Q&A pairs
```

## Data Formats

### Web Scraped Data Format

```json
{
  "metadata": {
    "source_url": "https://en.wikipedia.org/wiki/Tourism_in_Egypt",
    "title": "Tourism in Egypt",
    "scraped_at": "2024-07-24T10:30:00",
    "content_type": "wikipedia",
    "language": "en",
    "word_count": 1500,
    "paragraph_count": 25,
    "heading_count": 8
  },
  "content": {
    "title": "Tourism in Egypt",
    "paragraphs": ["...", "..."],
    "headings": [
      {"level": "h2", "text": "History"},
      {"level": "h3", "text": "Ancient Sites"}
    ]
  }
}
```

### PDF Extracted Data Format

```json
{
  "metadata": {
    "filename": "egypt_guide.pdf",
    "file_path": "/path/to/egypt_guide.pdf",
    "file_size_bytes": 2048576,
    "file_size_mb": 1.95,
    "extracted_at": "2024-07-24T10:30:00",
    "pdf_info": {
      "page_count": 15,
      "title": "Egypt Travel Guide",
      "author": "Tourism Board",
      "subject": "Travel Information"
    }
  },
  "text_extraction": {
    "method": "pymupdf",
    "pages": [...],
    "full_text": "...",
    "statistics": {
      "total_pages": 15,
      "total_words": 5000,
      "total_characters": 25000
    }
  },
  "layout_extraction": {
    "method": "pdfplumber",
    "pages": [...],
    "tables": [...],
    "statistics": {
      "total_pages": 15,
      "total_tables": 3,
      "total_images": 5
    }
  }
}
```

### Q&A Pairs Format

```json
{
  "metadata": {
    "created_at": "2024-07-24T10:30:00",
    "total_pairs": 150,
    "sources": ["https://en.wikipedia.org/wiki/Tourism_in_Egypt", "..."]
  },
  "qa_pairs": [
    {
      "question": "What are the main tourist attractions in Egypt?",
      "answer": "Egypt is home to some of the world's most famous ancient monuments...",
      "source": "https://en.wikipedia.org/wiki/Tourism_in_Egypt",
      "content_type": "wikipedia",
      "category": "attractions"
    }
  ]
}
```

## Configuration

### Customizing Web Scraping

Edit `src/data_collection/web_scraper.py` to:

- Add new Wikipedia pages to `wiki_pages` list
- Add new government sites to `gov_sites` list
- Add new travel guides to `travel_sites` list
- Modify scraping delay (`self.delay`)
- Change user agent headers

### Customizing PDF Processing

Edit `src/data_collection/pdf_extractor.py` to:

- Modify OCR settings (DPI, language)
- Change text extraction parameters
- Adjust table detection settings
- Customize image extraction options

### Customizing Q&A Generation

Edit `src/data_collection/data_collector.py` to:

- Add new question templates
- Modify answer extraction logic
- Change content categorization
- Adjust answer length limits

## Advanced Usage

### Custom Data Sources

To add custom data sources:

1. Create a new scraper class inheriting from `EgyptTourismScraper`
2. Implement custom scraping methods
3. Add to the main collector pipeline

### Batch Processing

For large-scale data collection:

```python
from data_collection import TourismDataCollector

collector = TourismDataCollector()

# Process multiple directories
for pdf_dir in ['pdfs1', 'pdfs2', 'pdfs3']:
    collector.collect_pdf_data(pdf_dir)
```

### Incremental Updates

To update existing data:

```bash
# Only scrape new web content
python scripts/run_data_collection.py --web-only

# Only process new PDFs
python scripts/run_data_collection.py --pdf-only --pdf-dir new_pdfs/

# Regenerate Q&A pairs with new data
python scripts/run_data_collection.py --qa-only
```

## Troubleshooting

### Common Issues

1. **Web Scraping Fails**
   - Check internet connection
   - Verify website accessibility
   - Adjust scraping delay
   - Check for rate limiting

2. **PDF Processing Errors**
   - Ensure PDF files are not corrupted
   - Check file permissions
   - Verify PDF format compatibility
   - Install required system dependencies (Tesseract for OCR)

3. **Memory Issues**
   - Process PDFs in smaller batches
   - Reduce image DPI for OCR
   - Use smaller text chunks

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

- Use multiprocessing for large PDF files
- Implement caching for web requests
- Use streaming for large text files
- Optimize image processing parameters

## Best Practices

1. **Respect Robots.txt**
   - Check website terms of service
   - Implement appropriate delays
   - Use proper user agents

2. **Data Quality**
   - Validate extracted text
   - Check for encoding issues
   - Verify metadata accuracy

3. **Storage Management**
   - Compress large files
   - Archive old data
   - Implement data versioning

4. **Error Handling**
   - Implement retry logic
   - Log all errors
   - Graceful degradation

## Integration with Training Pipeline

The collected data is ready for use in the training pipeline:

1. **Dataset Generation**: Use Q&A pairs for fine-tuning
2. **Data Validation**: Verify data quality before training
3. **Model Evaluation**: Test on collected data
4. **Continuous Updates**: Regularly refresh data sources

## Monitoring and Analytics

Track collection metrics:

- Number of sources processed
- Data quality scores
- Processing time
- Error rates
- Q&A pair generation statistics

Use the generated summary files to monitor collection performance and identify areas for improvement. 