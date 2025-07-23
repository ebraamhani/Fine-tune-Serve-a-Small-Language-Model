# Fine-tune-Serve-a-Small-Language-Model

A comprehensive project for fine-tuning and serving small language models using modern ML tools and best practices.

## 🚀 Quick Start

### 1. Setup Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Hugging Face Token

```bash
# Run the interactive setup script
python scripts/setup_huggingface.py
```

This will:
- Guide you through getting a Hugging Face token
- Test the token validity
- Save it to `.env` file
- Configure cache directories

### 3. Test Configuration

```bash
# Test Hugging Face setup
python scripts/test_huggingface.py
```

## 📁 Project Structure

```
Fine-tune-Serve-a-Small-Language-Model/
├── config/                    # Configuration files
│   ├── __init__.py
│   └── huggingface_config.py  # HF token management
├── data/                      # Raw and processed data
├── src/                       # Source code
│   ├── data_collection/       # Data collection scripts
│   ├── data_processing/       # Data preprocessing and cleaning
│   ├── dataset_generation/    # Dataset creation and formatting
│   ├── training/              # Model training scripts
│   ├── evaluation/            # Model evaluation and metrics
│   ├── deployment/            # Model serving and deployment
│   └── orchestration/         # Workflow orchestration
├── notebooks/                 # Jupyter notebooks for exploration
├── scripts/                   # Utility scripts
│   ├── setup_huggingface.py   # HF token setup
│   └── test_huggingface.py    # Configuration testing
├── .env                       # Environment variables (auto-generated)
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration

### Hugging Face Token

The project uses Hugging Face tokens for:
- Downloading pre-trained models
- Uploading fine-tuned models
- Accessing datasets
- Using gated models

To get your token:
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "fine-tuning-project")
4. Select "Write" role for full access
5. Copy the generated token

The token is automatically saved to `.env` file and configured in the environment.

### Environment Variables

Key environment variables:
- `HUGGINGFACE_TOKEN`: Your Hugging Face API token
- `HF_TOKEN`: Alternative token variable
- `HF_HOME`: Cache directory for models
- `TRANSFORMERS_CACHE`: Cache directory for transformers

## 🛠️ Key Tools

- **LangChain**: Framework for LLM applications
- **Hugging Face**: Model hub and transformers library
- **PyPDF2**: PDF processing
- **BeautifulSoup**: Web scraping
- **FastAPI**: API framework for model serving
- **DVC**: Data version control
- **Weights & Biases**: Experiment tracking
- **LoRA**: Low-rank adaptation for efficient fine-tuning

## 🔄 CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) includes:
- Python environment setup
- Dependency installation
- Code linting and formatting
- Basic testing
- Security checks

## 📚 Usage Examples

### Download a Model
```python
from transformers import AutoModel, AutoTokenizer

# Download model (requires token for gated models)
model = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
```

### Load a Dataset
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("squad", split="train")
```

### Use Configuration
```python
from config.huggingface_config import hf_config

# Check if logged in
if hf_config.is_logged_in():
    print("Ready to use Hugging Face!")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.