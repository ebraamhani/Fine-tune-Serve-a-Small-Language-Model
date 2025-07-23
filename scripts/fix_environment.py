#!/usr/bin/env python3
"""
Fix Environment Issues
Install missing packages and fix environment variable loading
"""

import os
import sys
import subprocess
from pathlib import Path

def install_missing_packages():
    """Install missing packages"""
    print("📦 Installing missing packages...")
    
    packages = [
        "datasets",
        "python-dotenv"  # For loading .env files
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent.parent / '.env'
    
    if env_file.exists():
        print(f"📄 Loading environment from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
                    print(f"✅ Loaded {key}")
    else:
        print("❌ .env file not found")

def test_environment():
    """Test if environment is properly set up"""
    print("\n🧪 Testing environment...")
    
    # Check token
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if token:
        print(f"✅ Token found: {token[:8]}...")
    else:
        print("❌ No token found in environment")
    
    # Test imports
    try:
        import datasets
        print(f"✅ datasets {datasets.__version__}")
    except ImportError:
        print("❌ datasets not available")
    
    try:
        import dotenv
        print(f"✅ python-dotenv available")
    except ImportError:
        print("❌ python-dotenv not available")

def main():
    """Main function"""
    print("🔧 Fixing Environment Issues")
    print("=" * 40)
    
    # Install missing packages
    install_missing_packages()
    
    # Load environment
    load_env_file()
    
    # Test environment
    test_environment()
    
    print("\n✅ Environment fix complete!")
    print("Run 'python scripts/test_huggingface.py' to verify everything works.")

if __name__ == "__main__":
    main() 