"""
Training Configuration for Egypt Tourism Assistant
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training configuration"""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # Larger, more capable model
    dataset_path: str = "data/datasets/splits"
    output_dir: str = "models/egypt_tourism_assistant"
    wandb_project: str = "egypt-tourism-assistant"
    
    # QLoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_length: int = 2048
    
    # Evaluation settings
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]

# Predefined configurations
CONFIGS = {
    "fast": TrainingConfig(
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        eval_steps=50,
        save_steps=100
    ),
    
    "standard": TrainingConfig(
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        eval_steps=100,
        save_steps=200
    ),
    
    "thorough": TrainingConfig(
        num_epochs=5,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        eval_steps=50,
        save_steps=100,
        warmup_steps=200
    ),
    
    "experimental": TrainingConfig(
        model_name="microsoft/DialoGPT-large",  # Larger model for testing
        num_epochs=2,
        batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        eval_steps=50,
        save_steps=100
    ),
    "very_fast": TrainingConfig(
        model_name="Qwen/Qwen1.5-0.5B-Chat",
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        lora_r=8,
        lora_alpha=16,
        eval_steps=200, # Set high to avoid frequent evaluation
        save_steps=400, # Set high to avoid frequent saving
        logging_steps=5
    ),
    
    "production": TrainingConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        num_epochs=5,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lora_r=32,
        lora_alpha=64,
        eval_steps=50,
        save_steps=100,
        warmup_steps=300,
        max_length=4096
    ),
    
    "high_quality": TrainingConfig(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        num_epochs=3,
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        lora_r=64,
        lora_alpha=128,
        eval_steps=25,
        save_steps=50,
        warmup_steps=500,
        max_length=4096
    )
}

def get_config(config_name: str = "standard") -> TrainingConfig:
    """Get training configuration by name"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name]

def list_configs():
    """List available configurations"""
    print("Available training configurations:")
    for name, config in CONFIGS.items():
        print(f"\n{name.upper()}:")
        print(f"  Model: {config.model_name}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  LoRA rank: {config.lora_r}")

if __name__ == "__main__":
    list_configs() 