#!/usr/bin/env python3
"""
Benchmark script to compare base model vs fine-tuned model performance.
"""

import json
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import get_config

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️  NLTK not available. Install with: pip install nltk")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  Rouge-score not available. Install with: pip install rouge-score")

try:
    from bert_score import score as bert_score_func
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("⚠️  BERTScore not available. Install with: pip install bert-score")

class ModelBenchmarker:
    """Benchmark class to compare model performance"""
    
    def __init__(self, model_name: str, fine_tuned_path: str = None):
        self.model_name = model_name
        self.fine_tuned_path = fine_tuned_path
        self.base_model = None
        self.base_tokenizer = None
        self.fine_tuned_model = None
        self.fine_tuned_tokenizer = None
        
    def load_models(self):
        """Load both base and fine-tuned models"""
        print(f"🔄 Loading models...")
        
        # Load base model
        print(f"📥 Loading base model: {self.model_name}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        
        # Load fine-tuned model if path provided
        if self.fine_tuned_path and Path(self.fine_tuned_path).exists():
            print(f"📥 Loading fine-tuned model: {self.fine_tuned_path}")
            self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(
                self.fine_tuned_path,
                trust_remote_code=True,
                padding_side="right"
            )
            if self.fine_tuned_tokenizer.pad_token is None:
                self.fine_tuned_tokenizer.pad_token = self.fine_tuned_tokenizer.eos_token
                
            self.fine_tuned_model = AutoModelForCausalLM.from_pretrained(
                self.fine_tuned_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        else:
            print("⚠️  Fine-tuned model not found, will only test base model")
            
        print("✅ Models loaded successfully")
    
    def generate_response(self, model, tokenizer, question: str, max_length: int = 200) -> str:
        """Generate response from a model"""
        # Format prompt based on model type and tokenizer capabilities
        try:
            if "qwen" in self.model_name.lower() and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant specializing in Egypt tourism information."},
                    {"role": "user", "content": question}
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        except Exception:
            # Fallback to simple format
            prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        input_ids_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_ids_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        if not BLEU_AVAILABLE:
            return 0.0
            
        # Tokenize
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        # Calculate BLEU
        smoothing = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if not ROUGE_AVAILABLE:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
    
    def calculate_bertscore(self, references: List[str], candidates: List[str]) -> List[float]:
        """Calculate BERTScore"""
        if not BERTSCORE_AVAILABLE:
            return [0.0] * len(references)
            
        try:
            P, R, F1 = bert_score_func(candidates, references, lang='en', verbose=False)
            return F1.tolist()
        except Exception as e:
            print(f"⚠️  BERTScore error: {e}")
            return [0.0] * len(references)
    
    def run_benchmark(self, benchmark_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Run the benchmark comparison"""
        print(f"🚀 Running benchmark with {len(benchmark_data)} questions...")
        
        results = {
            "metadata": {
                "model_name": self.model_name,
                "fine_tuned_path": self.fine_tuned_path,
                "num_questions": len(benchmark_data),
                "timestamp": datetime.now().isoformat(),
                "metrics_available": {
                    "bleu": BLEU_AVAILABLE,
                    "rouge": ROUGE_AVAILABLE,
                    "bertscore": BERTSCORE_AVAILABLE
                }
            },
            "base_model": {"responses": [], "metrics": {}},
            "fine_tuned_model": {"responses": [], "metrics": {}}
        }
        
        # Generate responses from base model
        print("🧠 Generating base model responses...")
        base_responses = []
        for i, qa in enumerate(benchmark_data):
            print(f"  Question {i+1}/{len(benchmark_data)}: {qa['question'][:50]}...")
            response = self.generate_response(self.base_model, self.base_tokenizer, qa['question'])
            base_responses.append({
                "question": qa['question'],
                "reference": qa['answer'],
                "generated": response
            })
        
        results["base_model"]["responses"] = base_responses
        
        # Generate responses from fine-tuned model if available
        if self.fine_tuned_model is not None:
            print("🤖 Generating fine-tuned model responses...")
            fine_tuned_responses = []
            for i, qa in enumerate(benchmark_data):
                print(f"  Question {i+1}/{len(benchmark_data)}: {qa['question'][:50]}...")
                response = self.generate_response(self.fine_tuned_model, self.fine_tuned_tokenizer, qa['question'])
                fine_tuned_responses.append({
                    "question": qa['question'],
                    "reference": qa['answer'],
                    "generated": response
                })
            
            results["fine_tuned_model"]["responses"] = fine_tuned_responses
        
        # Calculate metrics
        print("📊 Calculating metrics...")
        
        # Base model metrics
        base_metrics = self.calculate_metrics(base_responses)
        results["base_model"]["metrics"] = base_metrics
        
        # Fine-tuned model metrics
        if self.fine_tuned_model is not None:
            fine_tuned_metrics = self.calculate_metrics(fine_tuned_responses)
            results["fine_tuned_model"]["metrics"] = fine_tuned_metrics
        
        return results
    
    def calculate_metrics(self, responses: List[Dict[str, str]]) -> Dict[str, float]:
        """Calculate all metrics for a set of responses"""
        references = [r['reference'] for r in responses]
        candidates = [r['generated'] for r in responses]
        
        metrics = {}
        
        # BLEU scores
        bleu_scores = []
        for ref, cand in zip(references, candidates):
            bleu_scores.append(self.calculate_bleu(ref, cand))
        metrics["bleu"] = np.mean(bleu_scores)
        
        # ROUGE scores
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for ref, cand in zip(references, candidates):
            rouge_result = self.calculate_rouge(ref, cand)
            for key in rouge_scores:
                rouge_scores[key].append(rouge_result[key])
        
        for key in rouge_scores:
            metrics[key] = np.mean(rouge_scores[key])
        
        # BERTScore
        bert_scores = self.calculate_bertscore(references, candidates)
        metrics["bertscore"] = np.mean(bert_scores)
        
        return metrics
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results"""
        print("\n" + "="*60)
        print("📊 BENCHMARK RESULTS")
        print("="*60)
        
        print(f"Model: {results['metadata']['model_name']}")
        print(f"Questions: {results['metadata']['num_questions']}")
        print(f"Timestamp: {results['metadata']['timestamp']}")
        
        print("\n🧠 BASE MODEL METRICS:")
        base_metrics = results["base_model"]["metrics"]
        for metric, value in base_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        if "fine_tuned_model" in results and results["fine_tuned_model"]["responses"]:
            print("\n🤖 FINE-TUNED MODEL METRICS:")
            fine_tuned_metrics = results["fine_tuned_model"]["metrics"]
            for metric, value in fine_tuned_metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
            
            print("\n📈 IMPROVEMENT:")
            for metric in base_metrics.keys():
                if metric in fine_tuned_metrics:
                    improvement = fine_tuned_metrics[metric] - base_metrics[metric]
                    improvement_pct = (improvement / base_metrics[metric]) * 100 if base_metrics[metric] > 0 else 0
                    print(f"  {metric.upper()}: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # Print sample responses
        print("\n📝 SAMPLE RESPONSES:")
        for i, (base_resp, fine_resp) in enumerate(zip(
            results["base_model"]["responses"][:2],
            results.get("fine_tuned_model", {}).get("responses", [])[:2]
        )):
            print(f"\nQuestion {i+1}: {base_resp['question']}")
            print(f"Reference: {base_resp['reference'][:100]}...")
            print(f"Base: {base_resp['generated'][:100]}...")
            if fine_resp:
                print(f"Fine-tuned: {fine_resp['generated'][:100]}...")

def main():
    """Main benchmark function"""
    print("🏁 Starting Model Benchmark")
    print("="*50)
    
    # Load config
    config = get_config('very_fast')
    
    # Load benchmark dataset
    benchmark_file = Path("data/benchmark/egypt_tourism_benchmark.json")
    if not benchmark_file.exists():
        print("❌ Benchmark dataset not found. Run scripts/create_benchmark_dataset.py first")
        return
    
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker(
        model_name=config.model_name,
        fine_tuned_path=config.output_dir
    )
    
    # Load models
    benchmarker.load_models()
    
    # Run benchmark
    results = benchmarker.run_benchmark(benchmark_data["questions"])
    
    # Print results
    benchmarker.print_results(results)
    
    # Save results
    output_file = Path("data/benchmark/benchmark_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main() 