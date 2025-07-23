#!/usr/bin/env python3
"""
Test Hugging Face Configuration
Verify that Hugging Face is properly configured and working
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load environment at startup
load_env_file()

def test_huggingface_imports():
    """Test if Hugging Face packages can be imported"""
    print("🔍 Testing Hugging Face imports...")
    
    try:
        import transformers
        print(f"✅ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"✅ datasets {datasets.__version__}")
    except ImportError as e:
        print(f"❌ datasets: {e}")
        return False
    
    try:
        import huggingface_hub
        print(f"✅ huggingface_hub {huggingface_hub.__version__}")
    except ImportError as e:
        print(f"❌ huggingface_hub: {e}")
        return False
    
    try:
        import tokenizers
        print(f"✅ tokenizers {tokenizers.__version__}")
    except ImportError as e:
        print(f"❌ tokenizers: {e}")
        return False
    
    return True

def test_token_configuration():
    """Test token configuration"""
    print("\n🔑 Testing token configuration...")
    
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if not token:
        print("❌ No Hugging Face token found in environment variables")
        print("   Run: python scripts/setup_huggingface.py")
        return False
    
    print(f"✅ Token found: {token[:8]}...")
    return True

def test_huggingface_login():
    """Test Hugging Face login"""
    print("\n🔐 Testing Hugging Face login...")
    
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✅ Logged in as: {user}")
        return True
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

def test_model_download():
    """Test downloading a small model"""
    print("\n📥 Testing model download...")
    
    try:
        from transformers import AutoTokenizer
        
        # Try to download a small tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_auth_token=True)
        print("✅ Successfully downloaded GPT-2 tokenizer")
        return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

def test_dataset_access():
    """Test accessing a dataset"""
    print("\n📊 Testing dataset access...")
    
    try:
        from datasets import load_dataset
        
        # Try to load a small dataset
        dataset = load_dataset("squad", split="train[:10]")
        print(f"✅ Successfully loaded SQuAD dataset ({len(dataset)} samples)")
        return True
    except Exception as e:
        print(f"❌ Dataset access failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Hugging Face Configuration Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_huggingface_imports),
        ("Token Configuration", test_token_configuration),
        ("Login", test_huggingface_login),
        ("Model Download", test_model_download),
        ("Dataset Access", test_dataset_access),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Hugging Face is properly configured.")
    else:
        print("⚠️  Some tests failed. Please check your configuration.")
        print("\nTo fix issues:")
        print("1. Run: python scripts/setup_huggingface.py")
        print("2. Make sure you have a valid Hugging Face token")
        print("3. Check your internet connection")

if __name__ == "__main__":
    main() 