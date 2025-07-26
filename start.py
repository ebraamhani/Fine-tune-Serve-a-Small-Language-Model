#!/usr/bin/env python3
"""
🏺 Egypt Tourism Assistant - Ultra Simple Starter
Just run: python start.py
"""

import sys
import subprocess

def main():
    """Ultra simple starter"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = "help"
    
    if command == "api" or command == "start":
        print("🚀 Starting Egypt Tourism Assistant API...")
        print("📍 Visit: http://127.0.0.1:8000/docs")
        subprocess.run([sys.executable, "app.py"])
        
    elif command == "train":
        print("📊 Creating training data...")
        import requests
        try:
            response = requests.post("http://127.0.0.1:8000/train")
            print("✅ Training data created!")
        except:
            print("❌ Start API first: python start.py api")
            
    else:
        print("""
🏺 Egypt Tourism Assistant - Ultra Simple

Commands:
  python start.py api     # Start the API (DEFAULT)
  python start.py train   # Create training data
  
Quick Start:
  python start.py         # Starts API immediately
  
Then visit: http://127.0.0.1:8000/docs
Ask: "What are the top attractions in Egypt?"
        """)

if __name__ == "__main__":
    main() 