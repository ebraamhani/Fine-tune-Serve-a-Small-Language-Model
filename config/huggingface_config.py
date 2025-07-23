"""
Hugging Face Configuration
Handles token management and authentication for Hugging Face services
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class HuggingFaceConfig:
    """Configuration class for Hugging Face tokens and settings"""
    
    def __init__(self, token: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize Hugging Face configuration
        
        Args:
            token: Hugging Face API token
            cache_dir: Directory to cache models and datasets
        """
        self.token = token or self._get_token_from_env()
        self.cache_dir = cache_dir or self._get_cache_dir()
        self._setup_environment()
    
    def _get_token_from_env(self) -> Optional[str]:
        """Get token from environment variable"""
        return os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    def _get_cache_dir(self) -> str:
        """Get cache directory path"""
        cache_dir = os.getenv('HF_HOME') or os.getenv('TRANSFORMERS_CACHE')
        if not cache_dir:
            cache_dir = str(Path.home() / '.cache' / 'huggingface')
        return cache_dir
    
    def _setup_environment(self):
        """Set up environment variables for Hugging Face"""
        if self.token:
            os.environ['HUGGINGFACE_TOKEN'] = self.token
            os.environ['HF_TOKEN'] = self.token
            logger.info("Hugging Face token configured")
        else:
            logger.warning("No Hugging Face token found. Some features may be limited.")
        
        # Set cache directory
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def login(self, token: str):
        """Login to Hugging Face with token"""
        from huggingface_hub import login
        try:
            login(token=token)
            self.token = token
            self._setup_environment()
            logger.info("Successfully logged in to Hugging Face")
        except Exception as e:
            logger.error(f"Failed to login to Hugging Face: {e}")
            raise
    
    def is_logged_in(self) -> bool:
        """Check if user is logged in to Hugging Face"""
        try:
            from huggingface_hub import whoami
            whoami()
            return True
        except Exception:
            return False
    
    def get_token(self) -> Optional[str]:
        """Get current token"""
        return self.token
    
    def clear_token(self):
        """Clear stored token"""
        self.token = None
        if 'HUGGINGFACE_TOKEN' in os.environ:
            del os.environ['HUGGINGFACE_TOKEN']
        if 'HF_TOKEN' in os.environ:
            del os.environ['HF_TOKEN']
        logger.info("Hugging Face token cleared")

# Global instance
hf_config = HuggingFaceConfig() 