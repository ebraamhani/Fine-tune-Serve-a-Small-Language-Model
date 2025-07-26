#!/usr/bin/env python3
"""
🏺 Egypt Tourism Assistant - Ultra Simple FastAPI App
Everything in one clean file following FastAPI best practices
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup


# 🎯 Data Models
class TourismQuestion(BaseModel):
    question: str = Field(..., min_length=5, max_length=500, description="Your Egypt tourism question")
    temperature: float = Field(default=0.7, ge=0.1, le=1.0, description="Response creativity (0.1-1.0)")

class TourismAnswer(BaseModel):
    answer: str
    confidence: Optional[float] = None
    response_time: float
    suggestions: List[str] = []


# 🏺 Main FastAPI App
app = FastAPI(
    title="🏺 Egypt Tourism Assistant",
    description="AI-powered Egypt tourism Q&A assistant",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# 🧠 Global Variables
model = None
tokenizer = None
cache = {}
start_time = time.time()


# 📊 Data Pipeline Functions
def process_pdfs() -> List[dict]:
    """Extract Q&A from PDF files"""
    pdf_dir = Path("data/raw/pdf")
    if not pdf_dir.exists():
        return []
    
    qa_pairs = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            doc = fitz.open(pdf_file)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Simple Q&A generation
            if len(text) > 100:
                qa_pairs.extend([
                    {"question": "What are the main attractions in Egypt?", "answer": text[:300] + "..."},
                    {"question": "What should I know about Egypt travel?", "answer": text[300:600] + "..."}
                ])
        except:
            continue
    
    return qa_pairs


def get_quality_data() -> List[dict]:
    """High-quality Egypt tourism data"""
    return [
        {
            "question": "What currency is used in Egypt?",
            "answer": "The Egyptian Pound (EGP) is the official currency of Egypt. It's recommended to exchange money at banks or authorized dealers for the best rates."
        },
        {
            "question": "Do I need a visa to visit Egypt?",
            "answer": "Most visitors need a visa to enter Egypt. You can get a tourist visa on arrival at airports or apply online. Check with Egyptian consulates for your specific requirements."
        },
        {
            "question": "What are the top attractions in Egypt?",
            "answer": "Top attractions include the Pyramids of Giza, Sphinx, Egyptian Museum in Cairo, Valley of the Kings in Luxor, Abu Simbel temples, and the Red Sea resorts."
        },
        {
            "question": "When is the best time to visit Egypt?",
            "answer": "The best time to visit Egypt is from October to April when temperatures are cooler. Summer months (May-September) can be very hot, especially in southern regions."
        },
        {
            "question": "Is Egypt safe for tourists?",
            "answer": "Egypt is generally safe for tourists, especially in major tourist areas. Follow standard travel precautions, stay in tourist zones, and check current travel advisories."
        }
    ]


def create_training_data():
    """Create and save training dataset"""
    print("📊 Creating training data...")
    
    # Combine all data sources
    all_data = []
    all_data.extend(process_pdfs())
    all_data.extend(get_quality_data())
    
    # Ensure we have good data
    if len(all_data) < 3:
        all_data = get_quality_data()  # Fallback to quality data
    
    # Save training data
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / "tourism_data.json", "w", encoding="utf-8") as f:
        json.dump({"data": all_data}, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {len(all_data)} Q&A pairs")
    return True


# 🤖 AI Model Functions
async def load_model():
    """Load the tourism model"""
    global model, tokenizer
    
    try:
        model_path = "models/egypt_tourism_assistant"
        if not Path(model_path).exists():
            print("⚠️  No trained model found. Using base model.")
            model_path = "Qwen/Qwen1.5-0.5B-Chat"
        
        print("🔄 Loading model...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        
        print("✅ Model loaded!")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        model = None


def generate_answer(question: str, temperature: float = 0.7) -> dict:
    """Generate answer using AI model"""
    if model is None:
        # Fallback responses
        fallback_answers = {
            "currency": "Egypt uses the Egyptian Pound (EGP). Exchange at banks for best rates.",
            "visa": "Most visitors need a visa. Available on arrival or online.",
            "attractions": "Top sites: Pyramids, Sphinx, Egyptian Museum, Valley of Kings.",
            "weather": "Best time: October-April. Summer is very hot.",
            "safety": "Generally safe in tourist areas. Follow standard precautions."
        }
        
        for keyword, answer in fallback_answers.items():
            if keyword in question.lower():
                return {"answer": answer, "confidence": 0.8}
        
        return {"answer": "Egypt is a fascinating destination with rich history and culture. Please ask about specific topics like attractions, visa, currency, or weather.", "confidence": 0.6}
    
    try:
        # Format prompt
        messages = [{"role": "user", "content": f"As an Egypt tourism expert, answer this question: {question}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        confidence = min(len(response) / 100, 1.0)
        if any(word in response.lower() for word in ['egypt', 'pyramid', 'cairo']):
            confidence += 0.2
        
        return {"answer": response, "confidence": min(confidence, 1.0)}
        
    except Exception as e:
        return {"answer": f"I'm having trouble generating a response. Please try asking about Egypt's attractions, visa requirements, currency, or travel tips.", "confidence": 0.5}


def get_suggestions(question: str) -> List[str]:
    """Get follow-up suggestions"""
    if any(word in question.lower() for word in ['attraction', 'visit', 'see']):
        return ["How much time should I spend at the Pyramids?", "What's the best time to visit?"]
    elif 'visa' in question.lower():
        return ["How much does an Egypt visa cost?", "Can I get visa on arrival?"]
    elif 'currency' in question.lower():
        return ["Where to exchange money in Egypt?", "Do they accept credit cards?"]
    else:
        return ["What are top attractions in Egypt?", "Do I need a visa for Egypt?"]


# 🌐 API Routes
@app.on_event("startup")
async def startup():
    await load_model()


@app.get("/")
async def root():
    uptime = time.time() - start_time
    return {
        "message": "🏺 Egypt Tourism Assistant",
        "status": "running",
        "uptime_minutes": round(uptime / 60, 1),
        "endpoints": {
            "ask": "/ask",
            "docs": "/docs",
            "examples": "/examples"
        }
    }


@app.post("/ask", response_model=TourismAnswer)
async def ask_question(request: TourismQuestion):
    """Ask a tourism question about Egypt"""
    start = time.time()
    
    try:
        # Check cache
        cache_key = hashlib.md5(f"{request.question}:{request.temperature}".encode()).hexdigest()
        if cache_key in cache:
            cached = cache[cache_key].copy()
            cached['response_time'] = time.time() - start
            print(f"\n🤖 CACHED RESPONSE:")
            print(f"❓ Question: {request.question}")
            
            # Extract only the assistant's response (remove system/user prefixes)
            answer = cached['answer']
            if "assistant" in answer:
                # Find the assistant's response part
                assistant_start = answer.find("assistant")
                if assistant_start != -1:
                    # Get everything after "assistant"
                    assistant_response = answer[assistant_start + len("assistant"):].strip()
                    # Remove any remaining prefixes and clean up
                    if assistant_response.startswith("\n"):
                        assistant_response = assistant_response[1:].strip()
                    print(f"💬 Answer: {assistant_response}")
                else:
                    print(f"💬 Answer: {answer[:200]}...")
            else:
                print(f"💬 Answer: {answer[:200]}...")
                
            print(f"⏱️  Response Time: {cached['response_time']:.2f}s")
            print("-" * 50)
            return TourismAnswer(**cached)
        
        # Generate answer
        result = generate_answer(request.question, request.temperature)
        suggestions = get_suggestions(request.question)
        
        response_data = {
            "answer": result["answer"],
            "confidence": result.get("confidence"),
            "response_time": time.time() - start,
            "suggestions": suggestions
        }
        
        # Print response to console
        print(f"\n🤖 NEW RESPONSE:")
        print(f"❓ Question: {request.question}")
        
        # Extract only the assistant's response (remove system/user prefixes)
        answer = response_data['answer']
        if "assistant" in answer:
            # Find the assistant's response part
            assistant_start = answer.find("assistant")
            if assistant_start != -1:
                # Get everything after "assistant"
                assistant_response = answer[assistant_start + len("assistant"):].strip()
                # Remove any remaining prefixes and clean up
                if assistant_response.startswith("\n"):
                    assistant_response = assistant_response[1:].strip()
                print(f"💬 Answer: {assistant_response}")
            else:
                print(f"💬 Answer: {answer[:200]}...")
        else:
            print(f"💬 Answer: {answer[:200]}...")
            
        print(f"🎯 Confidence: {response_data.get('confidence', 'N/A')}")
        print(f"⏱️  Response Time: {response_data['response_time']:.2f}s")
        print(f"💡 Suggestions: {', '.join(suggestions[:3])}")
        print("-" * 50)
        
        # Cache result
        cache[cache_key] = response_data.copy()
        return TourismAnswer(**response_data)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("-" * 50)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/examples")
async def get_examples():
    """Example questions to try"""
    return {
        "attractions": [
            "What are the top attractions in Egypt?",
            "How much time should I spend at the Pyramids?",
            "What's special about Luxor?"
        ],
        "practical": [
            "Do I need a visa to visit Egypt?",
            "What currency should I bring?",
            "When is the best time to visit Egypt?"
        ],
        "culture": [
            "What should I wear in Egypt?",
            "What are important Egyptian customs?",
            "Is Egypt safe for tourists?"
        ]
    }


@app.get("/health")
async def health():
    """API health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cache_size": len(cache),
        "uptime_minutes": round((time.time() - start_time) / 60, 1)
    }


# 🚀 Training Endpoint
@app.post("/train")
async def train_model():
    """Create training data (simplified)"""
    try:
        success = create_training_data()
        return {
            "status": "success" if success else "failed",
            "message": "Training data created! Use external training script for model fine-tuning.",
            "data_location": "data/processed/tourism_data.json"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# 🎯 Run the app
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Egypt Tourism Assistant")
    print("📍 Visit: http://127.0.0.1:8000/docs")
    print("💬 Ask questions at: http://127.0.0.1:8000/ask")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error") 