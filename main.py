#!/usr/bin/env python3
"""
Main entry point for the Egypt Tourism Assistant application.
This script provides a command-line interface to run the training pipeline,
start the API server, or run the benchmark evaluation.
"""

import os
import sys
import click
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from src.training.qlora_trainer import QLoRATrainer
from src.api import app as fastapi_app
from scripts.run_benchmark import main as run_benchmark_main
from config.training_config import get_config

@click.group()
def cli():
    """Egypt Tourism Assistant: Fine-tuning, serving, and evaluating a specialized LLM."""
    pass

@cli.command()
def train():
    """Fine-tune the language model on the Egypt tourism dataset."""
    print("🚀 Starting model training pipeline...")
    try:
        config = get_config('very_fast')
        trainer = QLoRATrainer(
            model_name=config.model_name,
            dataset_path=config.dataset_path,
            output_dir=config.output_dir,
        )
        trainer.train()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")

@cli.command()
@click.option('--host', default='127.0.0.1', help='Host for the API server.')
@click.option('--port', default=8000, help='Port for the API server.')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development.')
def serve(host, port, reload):
    """Serve the fine-tuned model as a REST API."""
    print(f"🚀 Starting API server on http://{host}:{port}")
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

@cli.command()
def benchmark():
    """Evaluate the model's performance on the benchmark dataset."""
    print("📊 Running benchmark evaluation...")
    try:
        run_benchmark_main()
        print("✅ Benchmark completed successfully!")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    cli() 