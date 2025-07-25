#!/usr/bin/env python3
"""
Run QLoRA Training Pipeline
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training import QLoRATrainer

def main():
    """Run the training pipeline"""
    print("🚀 Starting QLoRA Training Pipeline")
    print("=" * 50)
    
    # Get fast configuration
    from config.training_config import get_config
    config = get_config('very_fast') # Using 'very_fast' config for a quick run
    
    # Initialize trainer with configuration parameters
    trainer = QLoRATrainer(
        model_name=config.model_name,
        dataset_path=config.dataset_path,
        output_dir=config.output_dir,
        wandb_project=config.wandb_project,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps
    )
    
    # Run training
    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
        
        # Test generation
        print("\n🧪 Testing model generation...")
        test_questions = [
            "What are the best tourist attractions in Egypt?",
            "Do I need a visa to visit Egypt?",
            "What is the best time to visit the Pyramids?",
            "Is it safe to travel to Egypt?",
            "What currency is used in Egypt?"
        ]
        
        for question in test_questions:
            response = trainer.generate_sample_response(question)
            print(f"\nQ: {question}")
            print(f"A: {response}")
            print("-" * 40)
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 