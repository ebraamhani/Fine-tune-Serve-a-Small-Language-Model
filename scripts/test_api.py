#!/usr/bin/env python3
"""
Test the Egypt Tourism Assistant API
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test an API endpoint"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        response.raise_for_status()
        return {
            "success": True,
            "status_code": response.status_code,
            "data": response.json(),
            "response_time": response.elapsed.total_seconds()
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        }

def main():
    """Test the API endpoints"""
    print("🧪 Testing Egypt Tourism Assistant API")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint...")
    result = test_endpoint("/")
    if result["success"]:
        print(f"✅ Root endpoint: {result['status_code']} - {result['data']['message']}")
    else:
        print(f"❌ Root endpoint failed: {result['error']}")
        return
    
    # Test 2: Health check
    print("\n2. Testing health endpoint...")
    result = test_endpoint("/health")
    if result["success"]:
        status = result["data"]["status"]
        model_loaded = result["data"]["model_loaded"]
        print(f"✅ Health: {status} - Model loaded: {model_loaded}")
    else:
        print(f"❌ Health endpoint failed: {result['error']}")
    
    # Test 3: Examples
    print("\n3. Testing examples endpoint...")
    result = test_endpoint("/examples")
    if result["success"]:
        examples = result["data"]["examples"]
        print(f"✅ Examples: {len(examples)} available")
        print(f"   Sample: {examples[0]}")
    else:
        print(f"❌ Examples endpoint failed: {result['error']}")
    
    # Test 4: Metrics (before predictions)
    print("\n4. Testing metrics endpoint...")
    result = test_endpoint("/metrics")
    if result["success"]:
        metrics = result["data"]
        print(f"✅ Metrics: {metrics['requests_total']} total requests")
        print(f"   Success rate: {metrics['success_rate']}%")
        print(f"   Uptime: {metrics['uptime_seconds']}s")
    else:
        print(f"❌ Metrics endpoint failed: {result['error']}")
    
    # Test 5: Predictions
    print("\n5. Testing prediction endpoint...")
    test_questions = [
        "What are the best tourist attractions in Egypt?",
        "Do I need a visa to visit Egypt?",
        "What currency is used in Egypt?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n5.{i} Question: {question}")
        
        prediction_data = {
            "question": question,
            "max_length": 150,
            "temperature": 0.7
        }
        
        result = test_endpoint("/predict", "POST", prediction_data)
        
        if result["success"]:
            data = result["data"]
            print(f"✅ Response time: {data['response_time']:.2f}s")
            print(f"   Answer: {data['answer'][:100]}...")
        else:
            print(f"❌ Prediction failed: {result['error']}")
    
    # Test 6: Final metrics
    print("\n6. Final metrics check...")
    result = test_endpoint("/metrics")
    if result["success"]:
        metrics = result["data"]
        print(f"✅ Final metrics:")
        print(f"   Total requests: {metrics['requests_total']}")
        print(f"   Successful: {metrics['requests_successful']}")
        print(f"   Failed: {metrics['requests_failed']}")
        print(f"   Success rate: {metrics['success_rate']}%")
        print(f"   Average response time: {metrics['average_response_time']}s")
    
    print("\n🎉 API testing completed!")

if __name__ == "__main__":
    main() 