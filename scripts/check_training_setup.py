#!/usr/bin/env python3
"""
Check Training Setup Status
"""

import json
from pathlib import Path

def check_training_setup():
    """Check the training setup status"""
    
    print("🔍 Training Setup Status Check")
    print("=" * 50)
    
    # Check dataset files
    print("\n📊 Dataset Status:")
    
    dataset_path = Path("data/datasets")
    if dataset_path.exists():
        print("✅ Dataset directory exists")
        
        # Check splits
        splits_path = dataset_path / "splits"
        if splits_path.exists():
            train_file = splits_path / "egypt_tourism_train.json"
            val_file = splits_path / "egypt_tourism_val.json"
            test_file = splits_path / "egypt_tourism_test.json"
            
            if train_file.exists():
                with open(train_file, 'r') as f:
                    train_data = json.load(f)
                print(f"✅ Train dataset: {len(train_data['data'])} samples")
            
            if val_file.exists():
                with open(val_file, 'r') as f:
                    val_data = json.load(f)
                print(f"✅ Validation dataset: {len(val_data['data'])} samples")
            
            if test_file.exists():
                with open(test_file, 'r') as f:
                    test_data = json.load(f)
                print(f"✅ Test dataset: {len(test_data['data'])} samples")
        else:
            print("❌ Splits directory not found")
    else:
        print("❌ Dataset directory not found")
    
    # Check model files
    print("\n🤖 Model Status:")
    
    model_path = Path("models/egypt_tourism_assistant")
    if model_path.exists():
        print("✅ Model directory exists")
        
        # Check for trained model files
        adapter_config = model_path / "adapter_config.json"
        adapter_model = model_path / "adapter_model.bin"
        training_config = model_path / "training_config.json"
        
        if adapter_config.exists():
            print("✅ LoRA adapter config found")
        if adapter_model.exists():
            print("✅ LoRA adapter weights found")
        if training_config.exists():
            print("✅ Training config found")
    else:
        print("ℹ️  Model directory not created yet (will be created during training)")
    
    # Check configuration
    print("\n⚙️ Configuration Status:")
    
    config_path = Path("config/training_config.py")
    if config_path.exists():
        print("✅ Training configuration exists")
        
        # Import and list configs
        try:
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from config.training_config import list_configs
            list_configs()
        except Exception as e:
            print(f"⚠️  Could not load configs: {e}")
    else:
        print("❌ Training configuration not found")
    
    # Check requirements
    print("\n📦 Requirements Status:")
    
    required_packages = [
        "transformers", "torch", "peft", "bitsandbytes", 
        "datasets", "wandb", "accelerate"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
    else:
        print("\n✅ All required packages installed")
    
    # Check GPU availability
    print("\n🖥️ Hardware Status:")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
        else:
            print("⚠️  CUDA not available - training will be slow on CPU")
    except ImportError:
        print("❌ PyTorch not installed")
    
    # Summary
    print("\n📋 Summary:")
    print("=" * 50)
    
    if dataset_path.exists() and splits_path.exists():
        print("✅ Dataset ready for training")
    else:
        print("❌ Dataset not ready - run dataset formatting first")
    
    if config_path.exists():
        print("✅ Configuration ready")
    else:
        print("❌ Configuration missing")
    
    if not missing_packages:
        print("✅ Dependencies ready")
    else:
        print("❌ Missing dependencies")
    
    print("\n🚀 Next Steps:")
    if dataset_path.exists() and not missing_packages:
        print("1. Run training: python scripts/run_training.py")
        print("2. Or use fast config: python -c 'from config.training_config import get_config; from src.training import QLoRATrainer; trainer = QLoRATrainer(**get_config(\"fast\").__dict__); trainer.train()'")
    else:
        print("1. Install missing dependencies")
        print("2. Format dataset: python src/dataset_generation/dataset_formatter.py")
        print("3. Run training: python scripts/run_training.py")

if __name__ == "__main__":
    check_training_setup() 