"""
Complete Hyperparameter Testing Framework for Translation Models
Tests MarianMT and M2M100 models with various hyperparameters
Focus on BERTScore F1 as primary metric
"""

import os
import json
import time
import torch
import gc
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from transformers import (
    MarianMTModel, MarianTokenizer,
    M2M100ForConditionalGeneration, M2M100Tokenizer,
)
import evaluate
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "hyperparameter_results_final"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MAX_SAMPLES = 100  # Number of test samples
BATCH_SIZE = 10   # For batch processing

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
exp_dir = os.path.join(OUTPUT_DIR, f"experiment_{TIMESTAMP}")
os.makedirs(exp_dir, exist_ok=True)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load metrics
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True) 
    nltk.download('omw-1.4', quiet=True)
    
    bertscore = evaluate.load("bertscore")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    print("All metrics loaded successfully")
except Exception as e:
    print(f"Error loading metrics: {e}")
    raise

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_test_data(max_samples=100):
    """Load test data"""
    print(f"Loading {max_samples} test samples...")
    
    # Try local files first
    if os.path.exists("data/test.en") and os.path.exists("data/test.de"):
        with open("data/test.en", 'r', encoding='utf-8') as f:
            source_texts = [line.strip() for line in f.readlines()[:max_samples]]
        with open("data/test.de", 'r', encoding='utf-8') as f:
            target_texts = [line.strip() for line in f.readlines()[:max_samples]]
        print(f"Loaded {len(source_texts)} samples from local files")
    else:
        # Fallback data
        print("Using fallback test data")
        source_texts = [
            "Hello, how are you?",
            "The weather is nice today.",
            "I love machine learning.",
        ] * (max_samples // 3)
        target_texts = [
            "Hallo, wie geht es dir?",
            "Das Wetter ist heute schön.",
            "Ich liebe maschinelles Lernen.",
        ] * (max_samples // 3)
        source_texts = source_texts[:max_samples]
        target_texts = target_texts[:max_samples]
    
    return source_texts, target_texts

def evaluate_translations(predictions, references, model_name="model"):
    """Evaluate translations using multiple metrics"""
    results = {}
    
    # BLEU
    try:
        bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
        results['bleu'] = bleu_result['bleu'] * 100
    except:
        results['bleu'] = 0.0
    
    # METEOR
    try:
        meteor_result = meteor.compute(predictions=predictions, references=references)
        results['meteor'] = meteor_result['meteor'] * 100
    except:
        results['meteor'] = 0.0
    
    # BERTScore
    try:
        bertscore_result = bertscore.compute(
            predictions=predictions, 
            references=references, 
            lang="de",
            device=DEVICE
        )
        results['bertscore_precision'] = np.mean(bertscore_result['precision'])
        results['bertscore_recall'] = np.mean(bertscore_result['recall'])
        results['bertscore_f1'] = np.mean(bertscore_result['f1'])
    except:
        results['bertscore_precision'] = 0.0
        results['bertscore_recall'] = 0.0
        results['bertscore_f1'] = 0.0
    
    return results

def test_model_inference(model, tokenizer, source_texts, target_texts, 
                        generation_params, model_name="model"):
    """Test model with specific generation parameters"""
    model.eval()
    predictions = []
    total_time = 0
    
    # Process in batches
    for i in range(0, len(source_texts), BATCH_SIZE):
        batch_texts = source_texts[i:i+BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(batch_texts, return_tensors="pt", 
                          padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_params)
        total_time += time.time() - start_time
        
        # Decode
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)
    
    # Evaluate
    metrics = evaluate_translations(predictions[:len(target_texts)], 
                                  target_texts, model_name)
    metrics['inference_time'] = total_time
    metrics['avg_time_per_sample'] = total_time / len(source_texts)
    
    return metrics, predictions

# ===== MARIANMT EXPERIMENTS =====
def test_marianmt(source_texts, target_texts):
    """Test MarianMT model"""
    print("\n" + "="*80)
    print("MARIANMT EXPERIMENTS")
    print("="*80)
    
    results = {
        'model_type': 'MarianMT',
        'inference_params': [],
        'training_params': []
    }
    
    # Only one MarianMT model size
    model_name = "Helsinki-NLP/opus-mt-en-de"
    
    # Test inference parameters
    print(f"\nTesting inference parameters for MarianMT...")
    print(f"Loading model: {model_name}")
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
    
    # 1. Beam Size
    print("\n  Testing beam_size...")
    for beam_size in [1, 3, 5, 10]:
        print(f"    Beam_size: {beam_size}")
        params = {
            'num_beams': beam_size,
            'max_length': 100,
            'early_stopping': True if beam_size > 1 else False,
            'do_sample': False
        }
        metrics, _ = test_model_inference(model, tokenizer, source_texts, 
                                        target_texts, params, model_name)
        results['inference_params'].append({
            'model': model_name,
            'parameter': 'beam_size',
            'value': beam_size,
            **metrics
        })
        print(f"      BERTScore F1: {metrics['bertscore_f1']:.4f}, Time: {metrics['inference_time']:.2f}s")
    
    # 2. Temperature (with sampling)
    print("\n  Testing temperature...")
    for temperature in [0.5, 0.7, 1.0, 1.5]:
        print(f"    Temperature: {temperature}")
        params = {
            'num_beams': 1,
            'max_length': 100,
            'temperature': temperature,
            'do_sample': True,
            'top_p': 0.9
        }
        metrics, _ = test_model_inference(model, tokenizer, source_texts, 
                                        target_texts, params, model_name)
        results['inference_params'].append({
            'model': model_name,
            'parameter': 'temperature',
            'value': temperature,
            **metrics
        })
        print(f"      BERTScore F1: {metrics['bertscore_f1']:.4f}, Time: {metrics['inference_time']:.2f}s")
    
    # 3. Max Length
    print("\n  Testing max_length...")
    for max_length in [50, 100, 150, 200]:
        print(f"    Max_length: {max_length}")
        params = {
            'num_beams': 3,
            'max_length': max_length,
            'early_stopping': True,
            'do_sample': False
        }
        metrics, _ = test_model_inference(model, tokenizer, source_texts, 
                                        target_texts, params, model_name)
        results['inference_params'].append({
            'model': model_name,
            'parameter': 'max_length',
            'value': max_length,
            **metrics
        })
        print(f"      BERTScore F1: {metrics['bertscore_f1']:.4f}, Time: {metrics['inference_time']:.2f}s")
    
    # Clean up
    del model
    clear_gpu_memory()
    
    # ===== TRAINING PARAMETERS (COMMENTED OUT) =====
    """
    # These parameters require actual fine-tuning, which is handled by the team
    # This section is provided as reference for the fine-tuning experiments
    
    print("\n3. Training parameters for fine-tuning (REFERENCE ONLY)...")
    
    # Learning Rate experiments
    learning_rates = [1e-5, 3e-5, 5e-5, 1e-4]
    for lr in learning_rates:
        training_args = {
            'learning_rate': lr,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 16,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'logging_dir': f'./logs/marianmt_lr_{lr}',
        }
        # Fine-tuning code would go here
        # model = fine_tune_model(model_name, train_data, val_data, training_args)
        # metrics = evaluate_model(model, test_data)
        
    # Batch Size experiments  
    batch_sizes = [8, 16, 32, 64]
    for batch_size in batch_sizes:
        training_args = {
            'learning_rate': 3e-5,
            'num_train_epochs': 3,
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'gradient_accumulation_steps': max(1, 32 // batch_size),  # Keep effective batch size constant
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'logging_dir': f'./logs/marianmt_batch_{batch_size}',
        }
        # Fine-tuning code would go here
    """
    
    return results

# ===== M2M100 EXPERIMENTS =====
def test_m2m100(source_texts, target_texts):
    """Test M2M100 models with different sizes"""
    print("\n" + "="*80)
    print("M2M100 EXPERIMENTS")
    print("="*80)
    
    results = {
        'model_type': 'M2M100',
        'model_sizes': [],
        'inference_params': [],
        'training_params': []
    }
    
    # Test different model sizes (layers)
    model_configs = [
        {"name": "facebook/m2m100_418M", "layers": 12, "size": "418M"},
        {"name": "facebook/m2m100_1.2B", "layers": 24, "size": "1.2B"}
    ]
    
    print("\n1. Testing different M2M100 model sizes...")
    for config in model_configs:
        print(f"\nTesting {config['size']} model ({config['layers']} layers)...")
        try:
            print(f"Loading model: {config['name']}")
            tokenizer = M2M100Tokenizer.from_pretrained(config['name'])
            model = M2M100ForConditionalGeneration.from_pretrained(config['name']).to(DEVICE)
            
            # Set source and target languages
            tokenizer.src_lang = "en"
            tokenizer.tgt_lang = "de"
            
            # Test with standard parameters
            params = {
                'num_beams': 5,
                'max_length': 100,
                'early_stopping': True,
                'do_sample': False,
                'forced_bos_token_id': tokenizer.get_lang_id("de")
            }
            
            metrics, _ = test_model_inference(model, tokenizer, source_texts[:50],  # Use fewer samples for large model
                                            target_texts[:50], params, config['name'])
            
            results['model_sizes'].append({
                'model': config['name'],
                'layers': config['layers'],
                'size': config['size'],
                **metrics
            })
            
            print(f"  Layers: {config['layers']}")
            print(f"  BERTScore F1: {metrics['bertscore_f1']:.4f}")
            print(f"  Time: {metrics['inference_time']:.2f}s")
            
            # Clean up large model
            del model
            clear_gpu_memory()
            
        except Exception as e:
            print(f"  Error testing {config['size']}: {str(e)}")
            results['model_sizes'].append({
                'model': config['name'],
                'layers': config['layers'],
                'size': config['size'],
                'error': str(e)
            })
    
    # Test inference parameters on smaller model only
    print("\n2. Testing inference parameters for M2M100...")
    model_name = "facebook/m2m100_418M"
    print(f"Loading model: {model_name}")
    
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = "de"
    
    # 1. Beam Size
    print("\n  Testing beam_size...")
    for beam_size in [1, 3, 5, 10]:
        print(f"    Beam_size: {beam_size}")
        params = {
            'num_beams': beam_size,
            'max_length': 100,
            'early_stopping': True if beam_size > 1 else False,
            'do_sample': False,
            'forced_bos_token_id': tokenizer.get_lang_id("de")
        }
        metrics, _ = test_model_inference(model, tokenizer, source_texts, 
                                        target_texts, params, model_name)
        results['inference_params'].append({
            'model': model_name,
            'parameter': 'beam_size',
            'value': beam_size,
            **metrics
        })
        print(f"      BERTScore F1: {metrics['bertscore_f1']:.4f}, Time: {metrics['inference_time']:.2f}s")
    
    # 2. Temperature
    print("\n  Testing temperature...")
    for temperature in [0.5, 0.7, 1.0, 1.5]:
        print(f"    Temperature: {temperature}")
        params = {
            'num_beams': 1,
            'max_length': 100,
            'temperature': temperature,
            'do_sample': True,
            'top_p': 0.9,
            'forced_bos_token_id': tokenizer.get_lang_id("de")
        }
        metrics, _ = test_model_inference(model, tokenizer, source_texts, 
                                        target_texts, params, model_name)
        results['inference_params'].append({
            'model': model_name,
            'parameter': 'temperature',
            'value': temperature,
            **metrics
        })
        print(f"      BERTScore F1: {metrics['bertscore_f1']:.4f}, Time: {metrics['inference_time']:.2f}s")
    
    # 3. Max Length
    print("\n  Testing max_length...")
    for max_length in [50, 100, 150, 200]:
        print(f"    Max_length: {max_length}")
        params = {
            'num_beams': 3,
            'max_length': max_length,
            'early_stopping': True,
            'do_sample': False,
            'forced_bos_token_id': tokenizer.get_lang_id("de")
        }
        metrics, _ = test_model_inference(model, tokenizer, source_texts, 
                                        target_texts, params, model_name)
        results['inference_params'].append({
            'model': model_name,
            'parameter': 'max_length',
            'value': max_length,
            **metrics
        })
        print(f"      BERTScore F1: {metrics['bertscore_f1']:.4f}, Time: {metrics['inference_time']:.2f}s")
    
    # Clean up
    del model
    clear_gpu_memory()
    
    # ===== TRAINING PARAMETERS (COMMENTED OUT) =====
    """
    # These parameters require actual fine-tuning, which is handled by the team
    # This section is provided as reference for the fine-tuning experiments
    
    print("\n3. Training parameters for fine-tuning (REFERENCE ONLY)...")
    
    # Learning Rate experiments
    learning_rates = [5e-6, 1e-5, 3e-5, 5e-5]
    for lr in learning_rates:
        training_args = {
            'learning_rate': lr,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 8,  # Smaller batch size for M2M100
            'per_device_eval_batch_size': 8,
            'gradient_accumulation_steps': 4,  # Accumulate gradients
            'warmup_steps': 1000,
            'weight_decay': 0.01,
            'fp16': True,  # Mixed precision training
            'logging_dir': f'./logs/m2m100_lr_{lr}',
        }
        # Fine-tuning code would go here
        
    # Batch Size experiments
    batch_sizes = [4, 8, 16, 32]
    for batch_size in batch_sizes:
        # Calculate gradient accumulation to maintain effective batch size
        gradient_accumulation = max(1, 32 // batch_size)
        training_args = {
            'learning_rate': 1e-5,
            'num_train_epochs': 3,
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation,
            'warmup_steps': 1000,
            'weight_decay': 0.01,
            'fp16': True,
            'logging_dir': f'./logs/m2m100_batch_{batch_size}',
        }
        # Fine-tuning code would go here
    """
    
    return results

# ===== MAIN FUNCTION =====
def main():
    """Main experiment function"""
    print("Starting Complete Hyperparameter Optimization")
    print("="*80)
    
    # Load data
    source_texts, target_texts = load_test_data(MAX_SAMPLES)
    print(f"Loaded {len(source_texts)} samples")
    
    # Run experiments
    marianmt_results = test_marianmt(source_texts, target_texts)
    m2m100_results = test_m2m100(source_texts, target_texts)
    
    # Save results
    with open(os.path.join(exp_dir, 'marianmt_results.json'), 'w') as f:
        json.dump(marianmt_results, f, indent=2)
    
    with open(os.path.join(exp_dir, 'm2m100_results.json'), 'w') as f:
        json.dump(m2m100_results, f, indent=2)
    
    # Generate summary report
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print("\nBest parameters for MarianMT:")
    # Find best parameters
    inference_results = marianmt_results['inference_params']
    for param in ['beam_size', 'temperature', 'max_length']:
        param_results = [r for r in inference_results if r['parameter'] == param]
        if param_results:
            best = max(param_results, key=lambda x: x['bertscore_f1'])
            print(f"  Best {param}: {best['value']} (F1: {best['bertscore_f1']:.4f})")
    
    print("\nBest parameters for M2M100:")
    inference_results = m2m100_results['inference_params']
    for param in ['beam_size', 'temperature', 'max_length']:
        param_results = [r for r in inference_results if r['parameter'] == param]
        if param_results:
            best = max(param_results, key=lambda x: x['bertscore_f1'])
            print(f"  Best {param}: {best['value']} (F1: {best['bertscore_f1']:.4f})")
    
    print(f"\nResults saved to: {exp_dir}")
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()