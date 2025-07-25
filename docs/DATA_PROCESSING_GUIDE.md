# Data Processing Guide

## Overview

The data processing pipeline for the Tourist QA Assistant ensures high-quality, deduplicated, and well-structured Q&A pairs for model training. This system handles text cleaning, quality checks, deduplication, language detection, and metadata enhancement.

## Key Features

### ✅ **Deduplication System**
- **Cross-source deduplication**: Removes duplicates between web and PDF sources
- **Similarity-based detection**: Uses SequenceMatcher with 85% similarity threshold
- **Content-type grouping**: Processes each source type separately for efficiency
- **Hash-based fast comparison**: MD5 hashing for exact duplicate detection

### ✅ **Quality Assurance**
- **Length validation**: Questions (10-200 chars), Answers (20-2000 chars)
- **Content validation**: Checks for empty, repetitive, or generic content
- **Source validation**: Ensures all pairs have valid sources
- **Quality scoring**: Assigns 0.0-1.0 quality scores based on multiple factors

### ✅ **Language Detection & Tagging**
- **Automatic detection**: Uses `langdetect` library for language identification
- **Multi-language support**: Currently supports English (en) and Arabic (ar)
- **Future-ready**: Language tags enable easy filtering for multi-language models
- **Consistent detection**: Seeded random generator for reproducible results

### ✅ **Metadata Enhancement**
- **Processing timestamps**: Tracks when each pair was processed
- **Length statistics**: Character and word counts
- **Source classification**: Encyclopedia, official, document, travel_guide
- **Quality metrics**: Individual quality scores for each pair

## Processing Results Summary

### 📊 **Performance Metrics**
- **Original pairs**: 157
- **Processed pairs**: 73
- **Removed pairs**: 84 (53.5% reduction)
- **Retention rate**: 46.5%

### 🎯 **Quality Metrics**
- **Average question length**: 34.9 characters
- **Average answer length**: 449.1 characters
- **Average quality score**: 1.000 (perfect quality)

### 🌐 **Language Distribution**
- **English (EN)**: 73 pairs (100%)

### 📚 **Content Type Distribution**
- **Wikipedia**: 63 pairs (86.3%)
- **Government sites**: 10 pairs (13.7%)

### 📂 **Category Distribution**
- **General info**: 58 pairs (79.5%)
- **Safety**: 2 pairs (2.7%)
- **Currency**: 2 pairs (2.7%)
- **Various specific topics**: 11 pairs (15.1%)

## File Structure

```
data/processed/
├── qa_pairs_processed.json          # Main processed dataset
├── qa_pairs_en.json                 # English-only dataset
├── metadata/
│   └── processing_summary.json      # Processing statistics
├── cleaned/                         # Text-cleaned data (future use)
├── deduplicated/                    # Deduplicated data (future use)
└── quality_checked/                 # Quality-checked data (future use)
```

## Usage

### Running the Full Pipeline

```bash
# Run complete data processing
python src/data_processing/data_processor.py
```

### Analyzing Results

```bash
# Analyze processing results
python scripts/analyze_processing.py
```

### Programmatic Usage

```python
from src.data_processing import TourismDataProcessor

# Initialize processor
processor = TourismDataProcessor(
    input_dir="data/raw",
    output_dir="data/processed"
)

# Run full processing
summary = processor.run_full_processing()
print(f"Processed {summary['processing_session']['processed_count']} pairs")
```

## Configuration

### Quality Thresholds

```python
# Adjustable in TourismDataProcessor.__init__()
self.min_answer_length = 20
self.max_answer_length = 2000
self.min_question_length = 10
self.max_question_length = 200
self.similarity_threshold = 0.85
```

### Language Detection

```python
# Supported languages
SUPPORTED_LANGUAGES = ['en', 'ar']

# Language detection settings
DetectorFactory.seed = 0  # For consistent results
```

## Processing Pipeline Steps

### 1. **Data Loading**
- Loads Q&A pairs from `data/raw/qa_pairs.json`
- Validates file structure and content

### 2. **Text Cleaning**
- Removes extra whitespace and special characters
- Normalizes quotes and punctuation
- Preserves meaningful content structure

### 3. **Quality Checking**
- Validates question and answer lengths
- Checks for empty or repetitive content
- Identifies generic or low-quality answers
- Validates source information

### 4. **Deduplication**
- Groups by content type (wikipedia, government_site, pdf_scanned)
- Uses hash-based exact duplicate detection
- Applies similarity-based fuzzy matching
- Removes 53% of Wikipedia duplicates and 41% of government site duplicates

### 5. **Language Detection**
- Detects language for questions and answers
- Assigns primary language tags
- Supports future multi-language filtering

### 6. **Metadata Enhancement**
- Adds processing timestamps
- Calculates quality scores
- Classifies source types
- Adds length and word count statistics

### 7. **Data Export**
- Saves processed data with metadata
- Creates language-specific files
- Generates processing summary

## Duplicate Analysis

### Wikipedia Content
- **Original**: 134 pairs
- **Processed**: 63 pairs
- **Removed**: 71 pairs (53.0%)
- **Reason**: High similarity in generated Q&A pairs from same source

### Government Sites
- **Original**: 17 pairs
- **Processed**: 10 pairs
- **Removed**: 7 pairs (41.2%)
- **Reason**: Similar questions about visa, entry requirements, etc.

### PDF Scanned Content
- **Original**: 6 pairs
- **Processed**: 0 pairs
- **Removed**: 6 pairs (100.0%)
- **Reason**: All failed quality checks due to generic content

## Quality Scoring System

### Score Calculation
```python
score = 1.0  # Base score

# Length penalties
if question_len < 20 or question_len > 150:
    score -= 0.2
if answer_len < 50 or answer_len > 1000:
    score -= 0.2

# Source quality bonus
if 'wikipedia' in source.lower():
    score += 0.1
elif 'gov' in source.lower():
    score += 0.1

# Content type bonus
if content_type == 'wikipedia':
    score += 0.1

return max(0.0, min(1.0, score))
```

### Quality Factors
- **Length appropriateness**: Questions and answers within optimal ranges
- **Source reliability**: Wikipedia and government sources get bonuses
- **Content uniqueness**: Avoids generic or repetitive content
- **Information completeness**: Ensures meaningful answers

## Future Enhancements

### 🔮 **Planned Features**
- **Arabic language support**: Enhanced Arabic text processing
- **Semantic deduplication**: Using embeddings for better similarity detection
- **Quality improvement**: ML-based quality scoring
- **Multi-language filtering**: Easy extraction of language-specific datasets
- **Incremental processing**: Process only new data
- **Quality feedback loop**: Learn from user feedback

### 🌍 **Multi-Language Support**
```python
# Future language detection
SUPPORTED_LANGUAGES = ['en', 'ar', 'fr', 'es', 'de']

# Language-specific processing
if pair['language_tag'] == 'ar':
    # Apply Arabic-specific cleaning
    pair['answer'] = arabic_text_cleaner(pair['answer'])
```

## Troubleshooting

### Common Issues

#### High Duplicate Rate
- **Cause**: Similar content from same sources
- **Solution**: Adjust similarity threshold or improve source diversity

#### Low Quality Scores
- **Cause**: Generic or short answers
- **Solution**: Improve data collection or adjust quality thresholds

#### Language Detection Errors
- **Cause**: Mixed language content or short text
- **Solution**: Increase minimum text length for detection

### Performance Optimization

#### Large Datasets
- **Batch processing**: Process in chunks for memory efficiency
- **Parallel processing**: Use multiprocessing for large datasets
- **Caching**: Cache language detection results

#### Memory Usage
- **Streaming**: Process files line by line
- **Garbage collection**: Clear intermediate results
- **Compression**: Use compressed file formats

## Best Practices

### ✅ **Do's**
- Run quality checks before training
- Monitor duplicate rates
- Validate language detection accuracy
- Keep processing logs for debugging
- Version control processed datasets

### ❌ **Don'ts**
- Skip quality validation
- Use unprocessed data for training
- Ignore duplicate warnings
- Mix languages without proper tagging
- Overwrite original data

## Integration with Training Pipeline

The processed data is ready for:
- **Fine-tuning**: High-quality, deduplicated pairs
- **Evaluation**: Quality-scored test sets
- **Multi-language models**: Language-tagged datasets
- **Domain-specific training**: Category-filtered subsets

## Monitoring and Maintenance

### Regular Checks
- **Quality metrics**: Monitor average quality scores
- **Duplicate rates**: Track deduplication effectiveness
- **Language distribution**: Ensure proper language detection
- **Processing time**: Optimize for large datasets

### Data Validation
- **Sample inspection**: Regularly check processed samples
- **Cross-validation**: Compare with manual quality assessment
- **Feedback integration**: Use user feedback to improve quality

---

*This data processing system ensures that only the highest quality, most relevant Q&A pairs are used for model training, significantly improving the performance and reliability of the Tourist QA Assistant.* 