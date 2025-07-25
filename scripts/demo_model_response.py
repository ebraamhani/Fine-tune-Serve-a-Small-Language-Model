#!/usr/bin/env python3
"""
Demo script to show real responses from our fine-tuned Egypt Tourism Assistant
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.qlora_trainer import QLoRATrainer
from config.training_config import get_config

def demo_model_responses():
    """Demonstrate real responses from our fine-tuned model"""
    print("🎯 Egypt Tourism Assistant - Live Demo")
    print("=" * 50)
    
    # Load configuration
    config = get_config('very_fast')
    
    # Initialize and load the model
    print("🤖 Loading fine-tuned Egypt Tourism Assistant...")
    trainer = QLoRATrainer(
        model_name=config.model_name,
        output_dir=config.output_dir,
    )
    
    try:
        trainer.load_model_and_tokenizer()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Demo questions
    demo_questions = [
        "What are the best tourist attractions in Egypt?",
        "Do I need a visa to visit Egypt?",
        "What currency is used in Egypt?",
        "What should I wear when visiting mosques?",
        "Is it safe to travel to Egypt right now?"
    ]
    
    print(f"\n🎪 Generating responses to {len(demo_questions)} tourism questions...")
    print("=" * 70)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("-" * 50)
        
        try:
            # Generate response
            response = trainer.generate_sample_response(
                instruction=question,
                max_length=150
            )
            
            print(f"🤖 Assistant: {response}")
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 Demo completed! This shows our fine-tuned model in action.")

if __name__ == "__main__":
    demo_model_responses() 