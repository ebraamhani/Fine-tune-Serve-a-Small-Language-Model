"""
QLoRA Trainer for Egypt Tourism Assistant
Fine-tunes Mistral-7B using QLoRA with PEFT
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable wandb completely
WANDB_AVAILABLE = False

class QLoRATrainer:
    """QLoRA trainer for fine-tuning Mistral-7B"""
    
    def __init__(self, 
                 model_name: str = "mistralai/Mistral-7B-v0.1",
                 dataset_path: str = "data/datasets/splits",
                 output_dir: str = "models/egypt_tourism_assistant",
                 wandb_project: str = "egypt-tourism-assistant",
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 num_epochs: int = 3,
                 batch_size: int = 4,
                 gradient_accumulation_steps: int = 4,
                 learning_rate: float = 2e-4,
                 warmup_steps: int = 100,
                 eval_steps: int = 100,
                 save_steps: int = 200,
                 logging_steps: int = 10):
        
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_project = wandb_project
        
        # Store configuration parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        
        # Store config for saving
        self.config_params = {
            "model_name": self.model_name,
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "wandb_project": self.wandb_project,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps
        }
        
        # QLoRA Configuration - adapt target modules based on model
        if "mistral" in self.model_name.lower():
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        elif "dialogpt" in self.model_name.lower():
            target_modules = [
                "c_attn", "c_proj", "c_fc", "c_proj"
            ]
        elif "qwen" in self.model_name.lower():
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        else:
            # Default for other models
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,  # Rank
            lora_alpha=self.lora_alpha,  # Alpha parameter for LoRA scaling
            lora_dropout=self.lora_dropout,  # Dropout probability for LoRA layers
            target_modules=target_modules
        )
        
        # BitsAndBytes Configuration for 4-bit quantization (disabled for compatibility)
        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        self.bnb_config = None
        
        # Training Configuration
        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
            fp16=False,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=3,
            report_to=None,  # Disable wandb completely
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0, # Set to 0 to avoid issues on Windows
            group_by_length=True,
            length_column_name="length"
        )
        
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.train_dataset = None
        
    def load_model_and_tokenizer(self):
        """Load Mistral-7B model and tokenizer with QLoRA setup"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model without quantization (simpler approach)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float32  # Use float32 for consistency
        )
        
        # No need for k-bit training preparation since we're not using quantization
        
        # Add LoRA adapters
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def load_datasets(self):
        """Load training dataset only"""
        logger.info("Loading training dataset...")
        
        # Load train dataset
        train_file = self.dataset_path / "egypt_tourism_train.json"
        if not train_file.exists():
            raise FileNotFoundError(f"Train dataset not found: {train_file}")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        logger.info(f"Loaded {len(train_data['data'])} training samples")
        
        # Convert to HuggingFace dataset
        self.train_dataset = Dataset.from_list(train_data['data'])
        
        logger.info("Training dataset loaded successfully")
    
    def format_instruction_prompt(self, example: Dict[str, Any]) -> str:
        """Format instruction prompt for training"""
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        # Use chat template for Qwen models
        if "qwen" in self.model_name.lower():
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant specializing in Egypt tourism information. Provide accurate, helpful, and detailed answers to questions about Egypt travel, attractions, culture, and practical information."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback to Alpaca format
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            return prompt

    def tokenize_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Tokenize a batch of examples"""
        # The 'examples' parameter is a dictionary of lists (e.g., {'instruction': [...], 'output': [...]})
        # We need to reconstruct each example and format it.
        num_examples = len(examples[next(iter(examples))])
        prompts = []
        for i in range(num_examples):
            # Create a dictionary for each individual example
            ex = {key: examples[key][i] for key in examples}
            prompts.append(self.format_instruction_prompt(ex))
        
        # Tokenize the whole batch of formatted prompts
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=2048,
            return_tensors=None
        )
        
        # Add length for grouping by the trainer
        tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]
        
        return tokenized

    def prepare_datasets(self):
        """Prepare datasets for training"""
        logger.info("Preparing datasets for training...")
        
        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        logger.info("Training dataset prepared successfully")
    

    
    def create_trainer(self) -> Trainer:
        """Create the trainer instance"""
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer (no evaluation)
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        return trainer
    
    def train(self):
        """Run the training process"""
        logger.info("Starting QLoRA training...")
        
        # Setup
        self.load_model_and_tokenizer()
        self.load_datasets()
        self.prepare_datasets()
        
        # Create trainer
        trainer = self.create_trainer()
        
        # Train
        logger.info("Training started...")
        trainer.train()
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training config
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_params, f, indent=2)
        
        logger.info(f"Training completed! Model saved to {self.output_dir}")
    
    def generate_sample_response(self, instruction: str, max_length: int = 512) -> str:
        """Generate a sample response using the trained model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Run load_model_and_tokenizer() first.")
        
        # Format prompt based on model type
        if "qwen" in self.model_name.lower():
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant specializing in Egypt tourism information."},
                {"role": "user", "content": instruction}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens, skipping the prompt
        input_ids_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_ids_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

def main():
    """Main function to run the training pipeline"""
    trainer = QLoRATrainer()
    trainer.train()
    
    # Test generation
    test_instruction = "What are the best tourist attractions in Egypt?"
    response = trainer.generate_sample_response(test_instruction)
    print(f"\nTest Generation:")
    print(f"Instruction: {test_instruction}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main() 