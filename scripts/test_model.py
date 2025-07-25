#!/usr/bin/env python3
"""
Test the fine-tuned model with sample questions.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.qlora_trainer import QLoRATrainer
from config.training_config import get_config

def main():
    """Main function to test the model"""
    print("🧪 Testing fine-tuned model...")
    print("=" * 50)
    
    # Get config used for training
    config = get_config('very_fast')

    # Initialize trainer to load the model
    trainer = QLoRATrainer(
        model_name=config.model_name,
        output_dir=config.output_dir,
    )
    
    # Load the fine-tuned model and tokenizer
    try:
        trainer.load_model_and_tokenizer()
        print("✅ Model loaded successfully from", config.output_dir)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please make sure you have run the training first (scripts/run_training.py)")
        return

    # Questions to test
    questions = [
        "What are the best tourist attractions in Egypt?",
        "Do I need a visa to visit Egypt?",
        "What is the best time to visit the Pyramids?",
        "Is it safe to travel to Egypt?",
        "What currency is used in Egypt?",
        "What should I wear when visiting religious sites?"
    ]

    for question in questions:
        print("-" * 50)
        print(f"🤔 Question: {question}")
        response = trainer.generate_sample_response(question)
        print(f"🤖 Answer: {response}")

if __name__ == "__main__":
    main() 