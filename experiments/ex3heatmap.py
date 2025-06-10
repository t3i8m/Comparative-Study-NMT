# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import numpy as np
import json
import pickle
from datetime import datetime
from transformers import (
    MarianMTModel, MarianTokenizer,
    M2M100ForConditionalGeneration, M2M100Tokenizer
)
from bert_score import score as bert_score

# Configuration
DATA_DIR = "/content/data"
OUTPUT_DIR = "/content/drive/MyDrive/nlp_subgrid_results"
CHECKPOINT_DIR = "/content/drive/MyDrive/nlp_subgrid_checkpoints"
MAX_SAMPLES = 50
BATCH_SIZE = 8

# Create directories
for dir_path in [OUTPUT_DIR, CHECKPOINT_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Load data function
def load_data(max_samples=MAX_SAMPLES):
    """Load test data for evaluation"""
    try:
        with open(f"{DATA_DIR}/test.en", 'r') as f:
            en_data = [line.strip() for line in f.readlines()[:max_samples]]
        with open(f"{DATA_DIR}/test.de", 'r') as f:
            de_data = [line.strip() for line in f.readlines()[:max_samples]]
        
        print(f"✓ Loaded {len(en_data)} test examples")
        return list(zip(en_data, de_data))
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create sample data for testing
        return [
            ("Hello world", "Hallo Welt"),
            ("How are you?", "Wie geht es dir?"),
            ("Good morning", "Guten Morgen"),
            ("Thank you", "Danke schön"),
            ("I love programming", "Ich liebe Programmieren")
        ] * 10

# Define parameter combinations for experiments
EXPERIMENT_CONFIGS = [
    {
        "name": "beam_maxlen_vs_beam_topp",
        "fixed_param": "beam_size",
        "fixed_values": [1, 3, 5],
        "row_params": ["max_length"],
        "row_values": [[50, 100, 150]],
        "col_params": ["top_p"],
        "col_values": [[0.7, 0.9, 0.95]]
    },
    {
        "name": "beam_temp_vs_beam_topp",
        "fixed_param": "beam_size",
        "fixed_values": [1, 3, 5],
        "row_params": ["temperature"],
        "row_values": [[0.7, 1.0, 1.3]],
        "col_params": ["top_p"],
        "col_values": [[0.7, 0.9, 0.95]]
    },
    {
        "name": "temp_beam_vs_temp_maxlen",
        "fixed_param": "temperature",
        "fixed_values": [0.7, 1.0, 1.3],
        "row_params": ["beam_size"],
        "row_values": [[1, 3, 5]],
        "col_params": ["max_length"],
        "col_values": [[50, 100, 150]]
    },
    {
        "name": "topp_beam_vs_topp_reppenalty",
        "fixed_param": "top_p",
        "fixed_values": [0.7, 0.9, 0.95],
        "row_params": ["beam_size"],
        "row_values": [[1, 3, 5]],
        "col_params": ["repetition_penalty"],
        "col_values": [[1.0, 1.2, 1.5]]
    },
    {
        "name": "maxlen_beam_vs_maxlen_temp",
        "fixed_param": "max_length",
        "fixed_values": [50, 100, 150],
        "row_params": ["beam_size"],
        "row_values": [[1, 3, 5]],
        "col_params": ["temperature"],
        "col_values": [[0.7, 1.0, 1.3]]
    },
    {
        "name": "beam_reppenalty_vs_beam_lenpenalty",
        "fixed_param": "beam_size",
        "fixed_values": [1, 3, 5],
        "row_params": ["repetition_penalty"],
        "row_values": [[1.0, 1.2, 1.5]],
        "col_params": ["length_penalty"],
        "col_values": [[0.8, 1.0, 1.2]]
    },
    {
        "name": "temp_topp_vs_temp_reppenalty",
        "fixed_param": "temperature",
        "fixed_values": [0.7, 1.0, 1.3],
        "row_params": ["top_p"],
        "row_values": [[0.7, 0.9, 0.95]],
        "col_params": ["repetition_penalty"],
        "col_values": [[1.0, 1.2, 1.5]]
    },
    {
        "name": "topp_maxlen_vs_topp_lenpenalty",
        "fixed_param": "top_p",
        "fixed_values": [0.7, 0.9, 0.95],
        "row_params": ["max_length"],
        "row_values": [[50, 100, 150]],
        "col_params": ["length_penalty"],
        "col_values": [[0.8, 1.0, 1.2]]
    },
    {
        "name": "reppenalty_beam_vs_reppenalty_temp",
        "fixed_param": "repetition_penalty",
        "fixed_values": [1.0, 1.2, 1.5],
        "row_params": ["beam_size"],
        "row_values": [[1, 3, 5]],
        "col_params": ["temperature"],
        "col_values": [[0.7, 1.0, 1.3]]
    },
    {
        "name": "lenpenalty_beam_vs_lenpenalty_topp",
        "fixed_param": "length_penalty",
        "fixed_values": [0.8, 1.0, 1.2],
        "row_params": ["beam_size"],
        "row_values": [[1, 3, 5]],
        "col_params": ["top_p"],
        "col_values": [[0.7, 0.9, 0.95]]
    },
    {
        "name": "beam_topk_vs_beam_temp",
        "fixed_param": "beam_size",
        "fixed_values": [1, 3, 5],
        "row_params": ["top_k"],
        "row_values": [[10, 50, 100]],
        "col_params": ["temperature"],
        "col_values": [[0.7, 1.0, 1.3]]
    }
]

class TranslationEvaluator:
    """Handles model loading and evaluation"""
    
    def __init__(self, model_type="marian"):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the translation model and tokenizer"""
        print(f"Loading {self.model_type} model...")
        
        if self.model_type == "marian":
            model_name = "Helsinki-NLP/opus-mt-en-de"
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
        else:  # m2m100
            model_name = "facebook/m2m100_418M"
            self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.tokenizer.src_lang = "en"
            self.tokenizer.tgt_lang = "de"
            
        self.model.eval()
        print(f"✓ Model loaded successfully")
        
    def translate_batch(self, texts, **generation_params):
        """Translate a batch of texts with given parameters"""
        try:
            # Prepare inputs
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, 
                                   truncation=True, max_length=512).to(self.device)
            
            # Set forced_bos_token_id for M2M100
            if self.model_type == "m2m100":
                generation_params["forced_bos_token_id"] = self.tokenizer.get_lang_id("de")
            
            # Generate translations
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)
            
            # Decode translations
            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return translations
            
        except Exception as e:
            print(f"Translation error: {e}")
            return [""] * len(texts)
    
    def evaluate_parameters(self, test_data, params):
        """Evaluate translation quality with given parameters"""
        sources = [pair[0] for pair in test_data]
        references = [pair[1] for pair in test_data]
        
        # Translate in batches
        translations = []
        for i in range(0, len(sources), BATCH_SIZE):
            batch = sources[i:i+BATCH_SIZE]
            batch_translations = self.translate_batch(batch, **params)
            translations.extend(batch_translations)
        
        # Calculate BERTScore
        if translations and all(t for t in translations):
            P, R, F1 = bert_score(translations, references, lang="de", 
                                 device=self.device, batch_size=BATCH_SIZE)
            return float(F1.mean())
        else:
            return 0.0

def create_subgrid_heatmap(evaluator, test_data, config, model_name):
    """Create heatmap with subgrids for different fixed parameter values"""
    
    fixed_param = config["fixed_param"]
    fixed_values = config["fixed_values"]
    
    # Create figure with subplots
    n_subgrids = len(fixed_values)
    fig, axes = plt.subplots(1, n_subgrids, figsize=(6*n_subgrids, 5))
    if n_subgrids == 1:
        axes = [axes]
    
    all_results = []
    
    for idx, fixed_val in enumerate(fixed_values):
        print(f"\nProcessing {fixed_param}={fixed_val}")
        
        # Create parameter grid
        row_values = config["row_values"][0]
        col_values = config["col_values"][0]
        
        results = np.zeros((len(row_values), len(col_values)))
        
        # Evaluate each combination
        for i, row_val in enumerate(row_values):
            for j, col_val in enumerate(col_values):
                # Build parameters
                params = {
                    fixed_param: fixed_val,
                    config["row_params"][0]: row_val,
                    config["col_params"][0]: col_val,
                    "max_new_tokens": 150,
                    "do_sample": True,
                    "num_beams": 1  # Default, will be overridden if beam_size is specified
                }
                
                # Adjust parameters based on what's being tested
                if "beam_size" in params:
                    params["num_beams"] = params.pop("beam_size")
                if "top_p" in params and params.get("num_beams", 1) > 1:
                    params["do_sample"] = False  # Beam search doesn't use sampling
                    params.pop("top_p", None)
                if "temperature" in params and params.get("num_beams", 1) > 1:
                    params["do_sample"] = False
                    params.pop("temperature", None)
                
                # Evaluate
                score = evaluator.evaluate_parameters(test_data, params)
                results[i, j] = score
                print(f"  {config['row_params'][0]}={row_val}, {config['col_params'][0]}={col_val}: {score:.3f}")
        
        # Create heatmap
        sns.heatmap(results, 
                   xticklabels=[f"{config['col_params'][0]}={v}" for v in col_values],
                   yticklabels=[f"{config['row_params'][0]}={v}" for v in row_values],
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=axes[idx], vmin=0, vmax=1)
        axes[idx].set_title(f"{fixed_param}={fixed_val}")
        
        all_results.append(results)
    
    # Overall title
    fig.suptitle(f"{model_name}: {config['name']}", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f"{model_name}_{config['name']}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # Save results data
    results_data = {
        "config": config,
        "results": all_results,
        "fixed_values": fixed_values
    }
    
    with open(output_path.replace('.png', '_data.pkl'), 'wb') as f:
        pickle.dump(results_data, f)
    
    return all_results

# Main execution
def run_experiments():
    """Run all experiments for both models"""
    
    # Load test data
    test_data = load_data()
    
    # Run experiments for each model
    for model_type in ["marian", "m2m100"]:
        print(f"\n{'='*60}")
        print(f"Running experiments for {model_type.upper()}")
        print(f"{'='*60}")
        
        # Initialize evaluator
        evaluator = TranslationEvaluator(model_type)
        evaluator.load_model()
        
        # Run each experiment configuration
        for i, config in enumerate(EXPERIMENT_CONFIGS):
            print(f"\nExperiment {i+1}/{len(EXPERIMENT_CONFIGS)}: {config['name']}")
            
            try:
                create_subgrid_heatmap(evaluator, test_data, config, model_type)
            except Exception as e:
                print(f"✗ Error in experiment: {e}")
                continue
        
        # Clear GPU memory
        del evaluator.model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {OUTPUT_DIR}")

# Run the experiments
if __name__ == "__main__":
    run_experiments()