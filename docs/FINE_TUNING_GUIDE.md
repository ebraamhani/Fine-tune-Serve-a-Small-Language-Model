# Fine-Tuning Guide

## Overview

This guide covers the fine-tuning setup for the Egypt Tourism Assistant using QLoRA (Quantized Low-Rank Adaptation) with Mistral-7B. The system uses PEFT (Parameter-Efficient Fine-Tuning) to efficiently adapt the base model to the tourism domain.

## 🎯 **Key Features**

### ✅ **QLoRA Implementation**
- **4-bit quantization** with BitsAndBytes for memory efficiency
- **LoRA adapters** with rank-16 for parameter-efficient training
- **Mistral-7B base model** for high-quality language understanding
- **PEFT integration** for efficient fine-tuning

### ✅ **Dataset Formatting**
- **Alpaca instruction format** for instruction-following training
- **Chat format** for conversational training
- **Enhanced instructions** with category-specific prompts
- **Train/validation/test splits** (80/10/10 ratio)

### ✅ **Training Infrastructure**
- **Weights & Biases integration** for experiment tracking
- **Configurable training parameters** for different scenarios
- **Automatic model saving** and checkpointing
- **Quality evaluation** during training

## 📊 **Dataset Statistics**

### **Formatted Dataset**
- **Original Q&A pairs**: 73 (processed)
- **Alpaca format samples**: 78 (with enhanced instructions)
- **Chat format samples**: 73
- **Train split**: 62 samples
- **Validation split**: 7 samples
- **Test split**: 9 samples

### **Data Quality**
- **Average quality score**: 1.000 (perfect)
- **Language distribution**: 100% English
- **Source types**: Wikipedia (86.3%), Government sites (13.7%)

## 🚀 **Quick Start**

### 1. **Format Dataset**
```bash
# Format Q&A pairs to instruction format
python src/dataset_generation/dataset_formatter.py
```

### 2. **Run Training**
```bash
# Standard training
python scripts/run_training.py

# Or with specific configuration
python -c "
from config.training_config import get_config
from src.training import QLoRATrainer
config = get_config('fast')
trainer = QLoRATrainer(**config.__dict__)
trainer.train()
"
```

### 3. **Test Model**
```python
from src.training import QLoRATrainer

trainer = QLoRATrainer()
trainer.load_model_and_tokenizer()

response = trainer.generate_sample_response(
    "What are the best tourist attractions in Egypt?"
)
print(response)
```

## ⚙️ **Configuration Options**

### **Available Configurations**

#### **Fast Training** (Quick testing)
- **Epochs**: 1
- **Batch size**: 2
- **Learning rate**: 3e-4
- **Use case**: Quick validation of setup

#### **Standard Training** (Recommended)
- **Epochs**: 3
- **Batch size**: 4
- **Learning rate**: 2e-4
- **Use case**: Production training

#### **Thorough Training** (Best quality)
- **Epochs**: 5
- **Batch size**: 2
- **Learning rate**: 1e-4
- **Use case**: Maximum quality

#### **Experimental** (Smaller model)
- **Model**: DialoGPT-medium
- **Epochs**: 2
- **Batch size**: 8
- **Use case**: Testing with limited resources

### **Custom Configuration**
```python
from config.training_config import TrainingConfig

config = TrainingConfig(
    model_name="mistralai/Mistral-7B-v0.1",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    lora_r=16,
    lora_alpha=32
)
```

## 📁 **File Structure**

```
src/
├── dataset_generation/
│   ├── dataset_formatter.py      # Dataset formatting
│   └── __init__.py
├── training/
│   ├── qlora_trainer.py          # QLoRA training
│   └── __init__.py
└── ...

config/
├── training_config.py            # Training configurations
└── ...

data/datasets/
├── alpaca/
│   └── egypt_tourism_alpaca.json # Alpaca format
├── chat/
│   └── egypt_tourism_chat.json   # Chat format
├── splits/
│   ├── egypt_tourism_train.json  # Training data
│   ├── egypt_tourism_val.json    # Validation data
│   └── egypt_tourism_test.json   # Test data
└── metadata/
    └── dataset_metadata.json     # Dataset statistics

models/
└── egypt_tourism_assistant/      # Trained model output
    ├── adapter_config.json       # LoRA configuration
    ├── adapter_model.bin         # LoRA weights
    ├── training_config.json      # Training parameters
    └── tokenizer/                # Tokenizer files
```

## 🔧 **Technical Details**

### **QLoRA Configuration**
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA scaling
    lora_dropout=0.1,        # Dropout rate
    target_modules=[         # Target layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

### **Quantization Settings**
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### **Training Arguments**
```python
training_args = TrainingArguments(
    output_dir="models/egypt_tourism_assistant",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=200,
    report_to="wandb"
)
```

## 📝 **Instruction Format**

### **Alpaca Format**
```json
{
    "instruction": "Answer the following question about Egypt tourism: What are the Pyramids?",
    "input": "",
    "output": "The Pyramids of Giza are ancient monumental structures...",
    "source": "https://en.wikipedia.org/wiki/Great_Pyramid_of_Giza",
    "category": "attractions",
    "quality_score": 1.0
}
```

### **Chat Format**
```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful AI assistant specializing in Egypt tourism..."
        },
        {
            "role": "user",
            "content": "What are the Pyramids?"
        },
        {
            "role": "assistant",
            "content": "The Pyramids of Giza are ancient monumental structures..."
        }
    ]
}
```

## 🧪 **Model Testing**

### **Sample Questions**
```python
test_questions = [
    "What are the best tourist attractions in Egypt?",
    "Do I need a visa to visit Egypt?",
    "What is the best time to visit the Pyramids?",
    "Is it safe to travel to Egypt?",
    "What currency is used in Egypt?",
    "How much does it cost to visit the Pyramids?",
    "What should I wear when visiting Egypt?",
    "Are there any health requirements for Egypt?",
    "What is the weather like in Egypt?",
    "How do I get from Cairo to Luxor?"
]
```

### **Generation Parameters**
```python
generation_params = {
    "max_length": 512,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}
```

## 📈 **Monitoring & Evaluation**

### **Weights & Biases Integration**
- **Project**: `egypt-tourism-assistant`
- **Metrics**: Training loss, validation loss, learning rate
- **Config tracking**: Model parameters, training settings
- **Artifact logging**: Model checkpoints, datasets

### **Training Metrics**
- **Loss curves**: Monitor training and validation loss
- **Learning rate**: Track learning rate scheduling
- **Gradient norms**: Monitor gradient flow
- **Memory usage**: Track GPU memory consumption

### **Evaluation Metrics**
- **Perplexity**: Language model quality
- **BLEU score**: Text generation quality
- **Human evaluation**: Manual quality assessment
- **Domain accuracy**: Tourism-specific knowledge

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Out of Memory (OOM)**
```python
# Reduce batch size
config.batch_size = 2
config.gradient_accumulation_steps = 8

# Use gradient checkpointing
training_args.gradient_checkpointing = True
```

#### **Slow Training**
```python
# Increase batch size if memory allows
config.batch_size = 8
config.gradient_accumulation_steps = 2

# Use mixed precision
training_args.fp16 = True
```

#### **Poor Quality Results**
```python
# Increase training epochs
config.num_epochs = 5

# Lower learning rate
config.learning_rate = 1e-4

# Increase LoRA rank
config.lora_r = 32
```

### **Performance Optimization**

#### **GPU Memory**
- **4-bit quantization**: Reduces memory by ~75%
- **Gradient checkpointing**: Trades compute for memory
- **Mixed precision**: Reduces memory usage
- **Gradient accumulation**: Simulates larger batch sizes

#### **Training Speed**
- **DataLoader workers**: Parallel data loading
- **Mixed precision**: Faster training
- **Group by length**: Reduces padding
- **Efficient tokenization**: Batch processing

## 🚀 **Deployment Preparation**

### **Model Export**
```python
# Save LoRA adapters
trainer.save_model()

# Save tokenizer
tokenizer.save_pretrained(output_dir)

# Create inference script
# See deployment/ directory
```

### **Model Loading**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "models/egypt_tourism_assistant")
```

## 📚 **Best Practices**

### ✅ **Do's**
- Start with fast configuration for testing
- Monitor training metrics closely
- Use validation set for early stopping
- Save multiple checkpoints
- Test model quality regularly
- Document training parameters

### ❌ **Don'ts**
- Train without validation monitoring
- Use too large learning rates
- Ignore memory usage warnings
- Skip model testing
- Overfit to small dataset
- Forget to backup checkpoints

## 🔮 **Future Enhancements**

### **Planned Features**
- **Multi-language support**: Arabic fine-tuning
- **Domain expansion**: Other tourism destinations
- **Advanced evaluation**: Automated quality metrics
- **Model compression**: Further optimization
- **Incremental training**: Continuous learning
- **A/B testing**: Model comparison framework

### **Research Directions**
- **Instruction tuning**: Better prompt engineering
- **Few-shot learning**: Minimal data adaptation
- **Knowledge distillation**: Smaller model training
- **Adversarial training**: Robustness improvement
- **Continual learning**: Incremental updates

---

*This fine-tuning setup provides an efficient and scalable approach to adapting large language models for the Egypt tourism domain, balancing quality, speed, and resource usage.* 