#!/usr/bin/env python3
"""
FastAPI Server for Egypt Tourism Assistant
Provides /predict, /health, and /metrics endpoints
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.qlora_trainer import QLoRATrainer
from config.training_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and metrics
model_instance: Optional[QLoRATrainer] = None
model_config = None
metrics = {
    "requests_total": 0,
    "requests_successful": 0,
    "requests_failed": 0,
    "average_response_time": 0.0,
    "model_loaded": False,
    "model_load_time": None,
    "startup_time": datetime.now().isoformat()
}

# Request/Response Models
class PredictionRequest(BaseModel):
    question: str = Field(..., description="Tourism question about Egypt", min_length=1, max_length=500)
    max_length: Optional[int] = Field(200, description="Maximum response length", ge=50, le=1000)
    temperature: Optional[float] = Field(0.7, description="Generation temperature", ge=0.1, le=2.0)

class PredictionResponse(BaseModel):
    question: str
    answer: str
    confidence: Optional[float] = None
    response_time: float
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None

class MetricsResponse(BaseModel):
    requests_total: int
    requests_successful: int
    requests_failed: int
    success_rate: float
    average_response_time: float
    model_loaded: bool
    model_load_time: Optional[str]
    startup_time: str
    uptime_seconds: float

async def load_model():
    """Load the fine-tuned model"""
    global model_instance, model_config, metrics
    
    start_time = time.time()
    logger.info("🤖 Loading fine-tuned Egypt tourism model...")
    
    try:
        # Get configuration
        model_config = get_config('very_fast')  # Use the same config as training
        
        # Initialize trainer
        model_instance = QLoRATrainer(
            model_name=model_config.model_name,
            output_dir=model_config.output_dir,
        )
        
        # Load the model
        model_instance.load_model_and_tokenizer()
        
        load_time = time.time() - start_time
        metrics["model_loaded"] = True
        metrics["model_load_time"] = datetime.now().isoformat()
        
        logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        metrics["model_loaded"] = False
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Egypt Tourism Assistant API")
    await load_model()
    yield
    # Shutdown
    logger.info("🛑 Shutting down Egypt Tourism Assistant API")

# Create FastAPI app
app = FastAPI(
    title="Egypt Tourism Assistant API",
    description="AI-powered assistant for Egypt tourism questions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add response time to headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Egypt Tourism Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        },
        "model_loaded": metrics["model_loaded"]
    }

@app.post("/predict", response_model=PredictionResponse, summary="Get tourism advice")
async def predict(request: PredictionRequest):
    """
    Get AI-powered answers to Egypt tourism questions
    
    - **question**: Your question about Egypt tourism
    - **max_length**: Maximum length of the response (50-1000)
    - **temperature**: Creativity of the response (0.1-2.0)
    """
    global metrics
    
    start_time = time.time()
    metrics["requests_total"] += 1
    
    try:
        # Check if model is loaded
        if not model_instance or not metrics["model_loaded"]:
            metrics["requests_failed"] += 1
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Generate response
        logger.info(f"📝 Processing question: {request.question[:50]}...")
        
        answer = model_instance.generate_sample_response(
            instruction=request.question,
            max_length=request.max_length
        )
        
        response_time = time.time() - start_time
        
        # Update metrics
        metrics["requests_successful"] += 1
        metrics["average_response_time"] = (
            (metrics["average_response_time"] * (metrics["requests_successful"] - 1) + response_time) 
            / metrics["requests_successful"]
        )
        
        # Prepare response
        response = PredictionResponse(
            question=request.question,
            answer=answer,
            response_time=response_time,
            model_info={
                "model_name": model_config.model_name,
                "output_dir": str(model_config.output_dir),
                "fine_tuned": True
            }
        )
        
        logger.info(f"✅ Response generated in {response_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        metrics["requests_failed"] += 1
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """
    Health check endpoint
    
    Returns the current status of the API and model
    """
    model_info = None
    if model_instance and metrics["model_loaded"]:
        model_info = {
            "model_name": model_config.model_name,
            "output_dir": str(model_config.output_dir),
            "fine_tuned": True,
            "parameters": "~468M (3.8M trainable)"
        }
    
    return HealthResponse(
        status="healthy" if metrics["model_loaded"] else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=metrics["model_loaded"],
        model_info=model_info
    )

@app.get("/metrics", response_model=MetricsResponse, summary="API metrics")
async def get_metrics():
    """
    Get API usage metrics and performance statistics
    """
    uptime = time.time() - datetime.fromisoformat(metrics["startup_time"]).timestamp()
    success_rate = (
        metrics["requests_successful"] / metrics["requests_total"] * 100 
        if metrics["requests_total"] > 0 else 0
    )
    
    return MetricsResponse(
        requests_total=metrics["requests_total"],
        requests_successful=metrics["requests_successful"],
        requests_failed=metrics["requests_failed"],
        success_rate=round(success_rate, 2),
        average_response_time=round(metrics["average_response_time"], 3),
        model_loaded=metrics["model_loaded"],
        model_load_time=metrics["model_load_time"],
        startup_time=metrics["startup_time"],
        uptime_seconds=round(uptime, 1)
    )

@app.get("/examples", summary="Example questions")
async def get_examples():
    """Get example questions you can ask the tourism assistant"""
    return {
        "examples": [
            "What are the best tourist attractions in Egypt?",
            "Do I need a visa to visit Egypt?",
            "What is the best time to visit the Pyramids?",
            "Is it safe to travel to Egypt?",
            "What currency is used in Egypt?",
            "What should I wear when visiting religious sites?",
            "How much should I tip in Egypt?",
            "What are the main airports in Egypt?",
            "Can I drink tap water in Egypt?",
            "What languages are spoken in Egypt?"
        ]
    } 