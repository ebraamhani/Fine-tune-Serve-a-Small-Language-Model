#!/usr/bin/env python3
"""
Dataset Validation Script for Egypt Tourism Assistant
Validates the structure and quality of training datasets
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def validate_dataset_file(file_path: Path) -> Dict[str, Any]:
    """Validate a single dataset file"""
    print(f"🔍 Validating: {file_path.name}")
    
    if not file_path.exists():
        return {"valid": False, "error": f"File not found: {file_path}"}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        return {"valid": False, "error": f"Error reading file: {e}"}
    
    # Check required top-level fields
    if 'metadata' not in data:
        return {"valid": False, "error": "Missing 'metadata' field"}
    if 'data' not in data:
        return {"valid": False, "error": "Missing 'data' field"}
    
    # Check metadata
    metadata = data['metadata']
    required_metadata = ['created_at', 'split', 'total_samples', 'format']
    for field in required_metadata:
        if field not in metadata:
            return {"valid": False, "error": f"Missing metadata field: {field}"}
    
    # Check data samples
    samples = data['data']
    if not samples:
        return {"valid": False, "error": "No data samples found"}
    
    # Validate sample structure
    required_sample_fields = ['instruction', 'output']
    optional_sample_fields = ['input', 'source', 'category', 'quality_score']
    
    for i, sample in enumerate(samples):
        # Check required fields
        for field in required_sample_fields:
            if field not in sample:
                return {"valid": False, "error": f"Sample {i} missing required field: {field}"}
        
        # Check field types
        if not isinstance(sample['instruction'], str):
            return {"valid": False, "error": f"Sample {i}: instruction must be string"}
        if not isinstance(sample['output'], str):
            return {"valid": False, "error": f"Sample {i}: output must be string"}
        
        # Check field lengths
        if len(sample['instruction']) < 10:
            return {"valid": False, "error": f"Sample {i}: instruction too short"}
        if len(sample['output']) < 10:
            return {"valid": False, "error": f"Sample {i}: output too short"}
    
    return {
        "valid": True,
        "sample_count": len(samples),
        "format": metadata.get('format'),
        "split": metadata.get('split'),
        "created_at": metadata.get('created_at')
    }

def main():
    """Main validation function"""
    print("🧪 Egypt Tourism Dataset Validation")
    print("=" * 50)
    
    # Define dataset paths
    dataset_dir = Path("data/datasets/splits")
    required_files = [
        "egypt_tourism_train.json",
        "egypt_tourism_val.json", 
        "egypt_tourism_test.json"
    ]
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    print(f"📁 Dataset directory: {dataset_dir}")
    print()
    
    # Validate each file
    results = {}
    all_valid = True
    
    for filename in required_files:
        file_path = dataset_dir / filename
        result = validate_dataset_file(file_path)
        results[filename] = result
        
        if result["valid"]:
            print(f"✅ {filename}")
            print(f"   Samples: {result['sample_count']}")
            print(f"   Format: {result['format']}")
            print(f"   Split: {result['split']}")
            print(f"   Created: {result['created_at']}")
        else:
            print(f"❌ {filename}")
            print(f"   Error: {result['error']}")
            all_valid = False
        print()
    
    # Summary
    print("=" * 50)
    print("📋 Validation Summary:")
    
    valid_count = sum(1 for r in results.values() if r["valid"])
    total_count = len(results)
    
    print(f"Files validated: {valid_count}/{total_count}")
    
    if all_valid:
        print("🎉 All datasets are valid!")
        
        # Calculate total samples
        total_samples = sum(r['sample_count'] for r in results.values() if r['valid'])
        print(f"📊 Total samples across all splits: {total_samples}")
        
        # Check for balanced splits
        train_samples = results.get("egypt_tourism_train.json", {}).get("sample_count", 0)
        val_samples = results.get("egypt_tourism_val.json", {}).get("sample_count", 0)
        test_samples = results.get("egypt_tourism_test.json", {}).get("sample_count", 0)
        
        if train_samples > 0 and val_samples > 0 and test_samples > 0:
            print(f"📈 Split distribution:")
            print(f"   Train: {train_samples} ({train_samples/total_samples*100:.1f}%)")
            print(f"   Validation: {val_samples} ({val_samples/total_samples*100:.1f}%)")
            print(f"   Test: {test_samples} ({test_samples/total_samples*100:.1f}%)")
        
        return 0
    else:
        print("❌ Some datasets failed validation")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 