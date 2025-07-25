#!/usr/bin/env python3
"""
Run the Egypt Tourism Assistant API Server
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """Start the FastAPI server"""
    print("🚀 Starting Egypt Tourism Assistant API Server")
    print("=" * 50)
    
    # Configuration
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"🌐 Server will start on: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔄 Reload mode: {reload}")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        "src.deployment.api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main() 