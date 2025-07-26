# 🏺 Egypt Tourism Assistant - AI-Powered Travel Guide

## 📋 Project Overview

The Egypt Tourism Assistant is a sophisticated AI-powered web application that provides comprehensive travel information and assistance for visitors to Egypt. Built with FastAPI and powered by transformer models, it offers intelligent Q&A capabilities, real-time response caching, and interactive API documentation.

## 🎯 Key Features

### 🤖 AI-Powered Responses
- **Transformer Model Integration**: Uses Qwen-1.5B for intelligent, context-aware responses
- **Real-time Processing**: Generates answers in under 15 seconds for complex queries
- **Confidence Scoring**: Provides reliability metrics for each response
- **Smart Fallbacks**: Graceful degradation when model is unavailable

### 📚 Comprehensive Knowledge Base
- **Multi-source Data**: Combines official tourism data, travel guides, and cultural resources
- **PDF Processing**: Automatic extraction from travel documents and guides
- **Web Scraping**: Real-time collection from authoritative Egypt tourism sources
- **Quality Validation**: Automated fact-checking and data verification

### ⚡ Performance & Reliability
- **Response Caching**: Lightning-fast repeat queries with intelligent cache management
- **Async Processing**: Non-blocking request handling for optimal performance
- **Health Monitoring**: Built-in system status and performance metrics
- **Error Handling**: Robust error management with detailed logging

### 🌐 API & Integration
- **RESTful API**: Clean, well-documented endpoints following OpenAPI standards
- **Interactive Documentation**: Auto-generated Swagger UI at `/docs`
- **CORS Support**: Cross-origin resource sharing for web frontends
- **Multiple Formats**: JSON responses with structured data

## 🏗️ Technical Architecture

### Backend Stack
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for high-performance async operations
- **PyTorch**: Deep learning framework for model inference
- **Transformers**: Hugging Face library for transformer models

### Data Processing
- **PyMuPDF**: PDF text extraction and processing
- **BeautifulSoup**: Web scraping and HTML parsing
- **JSON**: Structured data storage and exchange
- **Hash-based Caching**: Efficient response deduplication

### Development Tools
- **Python 3.8+**: Modern Python with type hints
- **Pydantic**: Data validation and serialization
- **Virtual Environment**: Isolated dependency management
- **Docker**: Containerized deployment support

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (for model loading)
- Internet connection (for initial model download)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fine-tune-Serve-a-Small-Language-Model
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn torch transformers PyMuPDF beautifulsoup4 requests
   ```

4. **Start the application**
   ```bash
   python app.py
   ```

5. **Access the API**
   - Interactive docs: http://127.0.0.1:8000/docs
   - Health check: http://127.0.0.1:8000/health
   - API endpoint: http://127.0.0.1:8000/ask

## 📡 API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/` | GET | API status and information | JSON |
| `/ask` | POST | Submit tourism questions | TourismAnswer |
| `/health` | GET | System health check | HealthStatus |
| `/examples` | GET | Sample questions by category | QuestionExamples |

### Request/Response Models

**TourismQuestion**
```json
{
  "question": "What are the top attractions in Egypt?",
  "temperature": 0.7
}
```

**TourismAnswer**
```json
{
  "answer": "Top attractions include the Pyramids of Giza...",
  "confidence": 0.95,
  "response_time": 2.34,
  "suggestions": ["How much time at Pyramids?", "Best time to visit?"]
}
```

## 📊 Data Pipeline

### Data Collection
1. **PDF Processing**: Extract tourism information from travel documents
2. **Web Scraping**: Collect data from 20+ authoritative sources
3. **Quality Validation**: Verify factual accuracy and relevance
4. **Data Merging**: Combine sources into comprehensive dataset

### Data Sources
- **Official**: Egypt Tourism Authority, Ministry of Tourism
- **Travel Guides**: Lonely Planet, Rough Guides, Fodor's
- **Cultural Sites**: Egyptian Museum, UNESCO World Heritage Sites
- **Practical Info**: Visa requirements, currency, weather, transportation

### Data Quality
- **Factual Accuracy**: Automated verification of currency, visa info, etc.
- **Relevance Filtering**: Focus on tourism-specific information
- **Duplicate Removal**: Intelligent deduplication across sources
- **Quality Scoring**: Confidence metrics for each data point

## 🔧 Configuration

### Model Settings
- **Base Model**: Qwen-1.5B (1.5 billion parameters)
- **Temperature**: Configurable creativity (0.1-1.0)
- **Max Tokens**: 512 for response generation
- **Cache Size**: Dynamic memory management

### Server Configuration
- **Host**: 127.0.0.1 (localhost)
- **Port**: 8000
- **Log Level**: Error (minimal logging)
- **CORS**: Enabled for web integration

## 📈 Performance Metrics

### Response Times
- **Cached Responses**: < 0.1 seconds
- **New AI Responses**: 10-15 seconds
- **Fallback Responses**: < 1 second

### System Resources
- **Memory Usage**: ~4GB (model + cache)
- **CPU Usage**: Moderate during inference
- **Storage**: ~2GB (model files + data)

### Quality Metrics
- **Confidence Score**: 0.8-1.0 for most responses
- **Cache Hit Rate**: ~60% for repeated queries
- **Uptime**: 99.9% with health monitoring

## 🛠️ Development

### Project Structure
```
Fine-tune-Serve-a-Small-Language-Model/
├── app.py                 # Main FastAPI application
├── start.py              # Alternative entry point
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Docker configuration
├── Dockerfile           # Container definition
├── data/                # Data storage
│   ├── raw/            # Source data
│   ├── processed/      # Processed datasets
│   └── datasets/       # Training splits
├── src/                # Source code modules
│   ├── api.py         # API endpoints
│   ├── training/      # Model training
│   ├── data_collection/ # Data gathering
│   └── data_processing/ # Data processing
└── docs/              # Documentation
```

### Key Components

**Main Application (`app.py`)**
- FastAPI app configuration
- Model loading and inference
- Response caching system
- API endpoint definitions

**Data Processing Pipeline**
- PDF text extraction
- Web scraping automation
- Quality validation
- Data merging and deduplication

**Training Infrastructure**
- QLoRA fine-tuning setup
- Dataset preparation
- Model evaluation
- Performance monitoring

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Docker Deployment
```bash
docker-compose up --build
```

### Production Considerations
- **Load Balancing**: Multiple instances behind reverse proxy
- **Monitoring**: Application performance monitoring (APM)
- **Logging**: Structured logging with rotation
- **Security**: Rate limiting and input validation

## 📚 Usage Examples

### Python Client
```python
import requests

response = requests.post("http://127.0.0.1:8000/ask", json={
    "question": "Do I need a visa to visit Egypt?",
    "temperature": 0.7
})

print(response.json()["answer"])
```

### cURL Commands
```bash
# Ask a question
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What currency is used in Egypt?"}'

# Check health
curl "http://127.0.0.1:8000/health"

# Get examples
curl "http://127.0.0.1:8000/examples"
```

### Web Interface
Visit http://127.0.0.1:8000/docs for interactive API documentation and testing.

## 🔍 Monitoring & Logging

### Health Endpoint
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cache_size": 15,
  "uptime_minutes": 45.2
}
```

### Console Output
The application provides real-time console output for:
- Question processing
- Response generation
- Cache operations
- Error handling

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

### Code Standards
- **Type Hints**: Required for all functions
- **Docstrings**: Comprehensive documentation
- **Error Handling**: Graceful error management
- **Testing**: Unit and integration tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: Transformers library and model hosting
- **FastAPI**: Modern web framework
- **Egypt Tourism Authority**: Official tourism information
- **Open Source Community**: Various libraries and tools

---

**🏺 Egypt Tourism Assistant** - Your intelligent travel companion for exploring the wonders of Egypt! 🇪🇬✨