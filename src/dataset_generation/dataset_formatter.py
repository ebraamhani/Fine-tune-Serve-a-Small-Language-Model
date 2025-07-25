"""
Dataset Formatter for Fine-tuning
Converts Q&A pairs to instruction format for Alpaca/Chat models
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetFormatter:
    """Formats Q&A pairs for fine-tuning"""
    
    def __init__(self, input_dir: str = "data/processed", output_dir: str = "data/datasets"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "alpaca").mkdir(exist_ok=True)
        (self.output_dir / "chat").mkdir(exist_ok=True)
        (self.output_dir / "splits").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
    
    def load_processed_data(self) -> List[Dict[str, Any]]:
        """Load processed Q&A pairs"""
        processed_file = self.input_dir / "qa_pairs_processed.json"
        
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed data not found: {processed_file}")
        
        logger.info(f"Loading processed data from {processed_file}")
        
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = data.get('qa_pairs', [])
        logger.info(f"Loaded {len(qa_pairs)} processed Q&A pairs")
        
        return qa_pairs
    
    def format_alpaca_instruction(self, qa_pair: Dict[str, Any]) -> Dict[str, str]:
        """Format Q&A pair to Alpaca instruction format"""
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        # Create instruction format
        instruction = f"Answer the following question about Egypt tourism: {question}"
        
        return {
            "instruction": instruction,
            "input": "",
            "output": answer,
            "source": qa_pair.get('source', ''),
            "category": qa_pair.get('category', ''),
            "quality_score": qa_pair.get('processing_metadata', {}).get('quality_score', 0)
        }
    
    def format_chat_instruction(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """Format Q&A pair to Chat format"""
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant specializing in Egypt tourism information. Provide accurate, helpful, and detailed answers to questions about Egypt travel, attractions, culture, and practical information."
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ],
            "source": qa_pair.get('source', ''),
            "category": qa_pair.get('category', ''),
            "quality_score": qa_pair.get('processing_metadata', {}).get('quality_score', 0)
        }
    
    def create_enhanced_instructions(self, qa_pair: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create enhanced instruction variations for better training"""
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        category = qa_pair.get('category', '')
        
        instructions = []
        
        # Base instruction
        base_instruction = f"Answer the following question about Egypt tourism: {question}"
        instructions.append({
            "instruction": base_instruction,
            "input": "",
            "output": answer,
            "source": qa_pair.get('source', ''),
            "category": category,
            "quality_score": qa_pair.get('processing_metadata', {}).get('quality_score', 0)
        })
        
        # Category-specific instructions
        if category == 'attractions':
            instructions.append({
                "instruction": f"Tell me about tourist attractions in Egypt: {question}",
                "input": "",
                "output": answer,
                "source": qa_pair.get('source', ''),
                "category": category,
                "quality_score": qa_pair.get('processing_metadata', {}).get('quality_score', 0)
            })
        elif category == 'safety':
            instructions.append({
                "instruction": f"Provide safety information for Egypt travel: {question}",
                "input": "",
                "output": answer,
                "source": qa_pair.get('source', ''),
                "category": category,
                "quality_score": qa_pair.get('processing_metadata', {}).get('quality_score', 0)
            })
        elif category == 'visa':
            instructions.append({
                "instruction": f"Explain visa requirements for Egypt: {question}",
                "input": "",
                "output": answer,
                "source": qa_pair.get('source', ''),
                "category": category,
                "quality_score": qa_pair.get('processing_metadata', {}).get('quality_score', 0)
            })
        elif category == 'culture':
            instructions.append({
                "instruction": f"Share information about Egyptian culture: {question}",
                "input": "",
                "output": answer,
                "source": qa_pair.get('source', ''),
                "category": category,
                "quality_score": qa_pair.get('processing_metadata', {}).get('quality_score', 0)
            })
        
        return instructions
    
    def create_train_val_test_splits(self, alpaca_data: List[Dict[str, Any]], 
                                   train_ratio: float = 0.8, 
                                   val_ratio: float = 0.1,
                                   test_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        """Create train/validation/test splits"""
        
        # Verify ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Shuffle data
        random.seed(42)  # For reproducibility
        shuffled_data = alpaca_data.copy()
        random.shuffle(shuffled_data)
        
        total_samples = len(shuffled_data)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        splits = {
            'train': shuffled_data[:train_end],
            'val': shuffled_data[train_end:val_end],
            'test': shuffled_data[val_end:]
        }
        
        logger.info(f"Created splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        
        return splits
    
    def save_formatted_data(self, alpaca_data: List[Dict[str, Any]], 
                          chat_data: List[Dict[str, Any]],
                          splits: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save formatted data to files"""
        
        # Save Alpaca format
        alpaca_file = self.output_dir / "alpaca" / "egypt_tourism_alpaca.json"
        with open(alpaca_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_samples': len(alpaca_data),
                    'format': 'alpaca',
                    'description': 'Egypt tourism Q&A pairs in Alpaca instruction format'
                },
                'data': alpaca_data
            }, f, indent=2, ensure_ascii=False)
        
        # Save Chat format
        chat_file = self.output_dir / "chat" / "egypt_tourism_chat.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_samples': len(chat_data),
                    'format': 'chat',
                    'description': 'Egypt tourism Q&A pairs in Chat format'
                },
                'data': chat_data
            }, f, indent=2, ensure_ascii=False)
        
        # Save splits
        for split_name, split_data in splits.items():
            split_file = self.output_dir / "splits" / f"egypt_tourism_{split_name}.json"
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'split': split_name,
                        'total_samples': len(split_data),
                        'format': 'alpaca'
                    },
                    'data': split_data
                }, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            'dataset_info': {
                'created_at': datetime.now().isoformat(),
                'total_original_pairs': len(alpaca_data),
                'alpaca_samples': len(alpaca_data),
                'chat_samples': len(chat_data),
                'train_samples': len(splits['train']),
                'val_samples': len(splits['val']),
                'test_samples': len(splits['test'])
            },
            'category_distribution': {},
            'quality_distribution': {
                'avg_quality_score': 0.0,
                'perfect_quality': 0,
                'high_quality': 0
            }
        }
        
        # Calculate distributions
        category_counts = {}
        quality_scores = []
        
        for item in alpaca_data:
            category = item.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            quality_score = item.get('quality_score', 0)
            quality_scores.append(quality_score)
        
        metadata['category_distribution'] = category_counts
        metadata['quality_distribution']['avg_quality_score'] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        metadata['quality_distribution']['perfect_quality'] = sum(1 for s in quality_scores if s == 1.0)
        metadata['quality_distribution']['high_quality'] = sum(1 for s in quality_scores if s >= 0.8)
        
        metadata_file = self.output_dir / "metadata" / "dataset_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved formatted data to {self.output_dir}")
    
    def run_full_formatting(self, use_enhanced_instructions: bool = True) -> Dict[str, Any]:
        """Run complete dataset formatting pipeline"""
        logger.info("Starting dataset formatting pipeline...")
        
        # Load processed data
        qa_pairs = self.load_processed_data()
        
        # Format to Alpaca instruction format
        logger.info("Formatting to Alpaca instruction format...")
        alpaca_data = []
        chat_data = []
        
        for qa_pair in qa_pairs:
            # Basic Alpaca format
            alpaca_item = self.format_alpaca_instruction(qa_pair)
            alpaca_data.append(alpaca_item)
            
            # Chat format
            chat_item = self.format_chat_instruction(qa_pair)
            chat_data.append(chat_item)
            
            # Enhanced instructions if enabled
            if use_enhanced_instructions:
                enhanced_items = self.create_enhanced_instructions(qa_pair)
                # Add enhanced items (skip the first one as it's already added)
                alpaca_data.extend(enhanced_items[1:])
        
        logger.info(f"Created {len(alpaca_data)} Alpaca format samples")
        logger.info(f"Created {len(chat_data)} Chat format samples")
        
        # Create splits
        logger.info("Creating train/validation/test splits...")
        splits = self.create_train_val_test_splits(alpaca_data)
        
        # Save formatted data
        logger.info("Saving formatted data...")
        self.save_formatted_data(alpaca_data, chat_data, splits)
        
        # Create summary
        summary = {
            'formatting_session': {
                'started_at': datetime.now().isoformat(),
                'original_pairs': len(qa_pairs),
                'alpaca_samples': len(alpaca_data),
                'chat_samples': len(chat_data),
                'enhanced_instructions': use_enhanced_instructions
            },
            'splits': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test'])
            }
        }
        
        logger.info("Dataset formatting completed!")
        return summary

if __name__ == "__main__":
    formatter = DatasetFormatter()
    summary = formatter.run_full_formatting(use_enhanced_instructions=True)
    print(json.dumps(summary, indent=2)) 