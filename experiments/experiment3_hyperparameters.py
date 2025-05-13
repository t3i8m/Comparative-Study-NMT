"""
Experiment 3: Hyperparameter Comparison for MarianMT Model

This script tests how different hyperparameters affect the performance of MarianMT
translation model. Specifically, it evaluates the impact of learning rate, batch size,
and other parameters on translation quality using BLEU score.

"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import nltk
import sacrebleu
import json
import time
from transformers import (
    MarianMTModel, 
    MarianTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
from datasets import Dataset
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project-specific modules
try:
    from transformers_models.marian.marianMT import MarianMt
except ImportError:
    print("WARNING: Could not import MarianMt class from project. Using direct implementation.")
    # Fallback implementation
    class MarianMt:
        def __init__(self, model_name):
            self.model_name = model_name
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            
        def get_model(self):
            return self.model
            
        def get_tokenizer(self):
            return self.tokenizer
            
        def tokenize_str(self, text):
            return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Download NLTK data if needed
nltk.download('punkt', quiet=True)

# Constants
DATA_DIR = "data"  # Adjust if necessary
OUTPUT_DIR = "experiment3_results"
MAX_SAMPLES = 5000  # Limit sample size for faster experimentation

def ensure_dir(directory):
    """Ensure that a directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(data_dir=DATA_DIR, max_samples=MAX_SAMPLES):
    """
    Load training and test data from files.
    
    Args:
        data_dir: Directory containing the data files
        max_samples: Maximum number of samples to use (for faster experimentation)
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print(f"Loading data from {data_dir}...")
    translation_pairs_en2de = {"en": [], "de": []}
    
    try:
        # Load training data
        train_en_path = os.path.join(data_dir, "train.en")
        train_de_path = os.path.join(data_dir, "train.de")
        
        if not (os.path.exists(train_en_path) and os.path.exists(train_de_path)):
            raise FileNotFoundError(f"Training files not found in {data_dir}. Run data downloaders first.")
            
        with open(train_en_path, encoding="utf-8") as f_en, \
             open(train_de_path, encoding="utf-8") as f_de:
            for i, (en, de) in enumerate(zip(f_en, f_de)):
                if i >= max_samples:  # Limit for faster experimentation
                    break
                translation_pairs_en2de["en"].append(en.strip())
                translation_pairs_en2de["de"].append(de.strip())
        
        print(f"Loaded {len(translation_pairs_en2de['en'])} training examples")
        
        # Create and split dataset
        dataset = Dataset.from_dict(translation_pairs_en2de)
        dataset = dataset.train_test_split(test_size=0.1)
        
        return dataset['train'], dataset['test']
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(model, train_data, val_data):
    """
    Preprocess data for training.
    
    Args:
        model: MarianMt model instance
        train_data: Training dataset
        val_data: Validation dataset
        
    Returns:
        Tuple of (tokenized_train, tokenized_val)
    """
    print("Preprocessing data...")
    
    def preprocess(example):
        inputs = model.tokenize_str(example["en"])
        targets = model.tokenize_str(example["de"])
        inputs["labels"] = targets["input_ids"]
        return inputs
    
    tokenized_train = train_data.map(preprocess, batched=True)
    tokenized_val = val_data.map(preprocess, batched=True)
    
    return tokenized_train, tokenized_val

def compute_metrics(eval_preds, tokenizer):
    """
    Calculate evaluation metrics for model predictions.
    
    Args:
        eval_preds: Tuple of (predictions, labels)
        tokenizer: Tokenizer to decode predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    preds, labels = eval_preds
    
    # Replace -100 (padding token id in labels) with tokenizer's pad token id
    labels = labels.copy()
    labels[labels == -100] = tokenizer.pad_token_id
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    
    return {
        "bleu": bleu.score
    }

def run_hyperparameter_experiment():
    """
    Run experiments testing different hyperparameters.
    
    Returns:
        List of dictionaries with experiment results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(OUTPUT_DIR, f"experiment_{timestamp}")
    ensure_dir(experiment_dir)
    
    print(f"Results will be saved to: {experiment_dir}")
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading base model...")
    base_model_name = "Helsinki-NLP/opus-mt-en-de"
    base_model = MarianMt(base_model_name)
    
    # Load data
    train_data, val_data = load_data()
    tokenized_train, tokenized_val = preprocess_data(base_model, train_data, val_data)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(base_model.get_tokenizer(), model=base_model.get_model())
    
    results = []
    
    # Experiment 1: Learning Rate
    learning_rates = [1e-5, 5e-5, 1e-4]
    for lr in learning_rates:
        experiment_name = f"learning_rate_{lr}"
        print(f"\n{'='*50}\nExperiment: Learning Rate = {lr}\n{'='*50}")
        
        # Create a new model instance for each experiment
        model = MarianMt(base_model_name)
        
        # Set training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(experiment_dir, f"model_lr{lr}"),
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=lr,
            num_train_epochs=2,  # Reduced for faster experimentation
            eval_strategy="epoch",  # Changed from evaluation_strategy
            save_strategy="epoch",
            predict_with_generate=True,
            logging_dir=os.path.join(experiment_dir, f"logs_lr{lr}"),
            logging_steps=100,
            save_total_limit=1,
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model.get_model(),
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=model.get_tokenizer(),
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, model.get_tokenizer())
        )
        
        try:
            # Train model
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            # Evaluate model
            eval_metrics = trainer.evaluate()
            
            # Save results
            result = {
                "experiment": "learning_rate",
                "value": lr,
                "bleu": eval_metrics["eval_bleu"],
                "eval_loss": eval_metrics["eval_loss"],
                "train_loss": train_result.training_loss,
                "training_time": training_time
            }
            results.append(result)
            
            print(f"Learning Rate {lr} results:")
            print(f"  BLEU Score: {eval_metrics['eval_bleu']:.2f}")
            print(f"  Evaluation Loss: {eval_metrics['eval_loss']:.4f}")
            print(f"  Training Loss: {train_result.training_loss:.4f}")
            print(f"  Training Time: {training_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error in experiment with learning rate {lr}: {e}")
    
    # Experiment 2: Batch Size
    batch_sizes = [8, 16, 32]
    for bs in batch_sizes:
        experiment_name = f"batch_size_{bs}"
        print(f"\n{'='*50}\nExperiment: Batch Size = {bs}\n{'='*50}")
        
        # Create a new model instance for each experiment
        model = MarianMt(base_model_name)
        
        # Set training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(experiment_dir, f"model_bs{bs}"),
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            learning_rate=5e-5,  # Use default learning rate
            num_train_epochs=2,  # Reduced for faster experimentation
            eval_strategy="epoch",  # Changed from evaluation_strategy
            save_strategy="epoch",
            predict_with_generate=True,
            logging_dir=os.path.join(experiment_dir, f"logs_bs{bs}"),
            logging_steps=100,
            save_total_limit=1,
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model.get_model(),
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=model.get_tokenizer(),
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, model.get_tokenizer())
        )
        
        try:
            # Train model
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            # Evaluate model
            eval_metrics = trainer.evaluate()
            
            # Save results
            result = {
                "experiment": "batch_size",
                "value": bs,
                "bleu": eval_metrics["eval_bleu"],
                "eval_loss": eval_metrics["eval_loss"],
                "train_loss": train_result.training_loss,
                "training_time": training_time
            }
            results.append(result)
            
            print(f"Batch Size {bs} results:")
            print(f"  BLEU Score: {eval_metrics['eval_bleu']:.2f}")
            print(f"  Evaluation Loss: {eval_metrics['eval_loss']:.4f}")
            print(f"  Training Loss: {train_result.training_loss:.4f}")
            print(f"  Training Time: {training_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error in experiment with batch size {bs}: {e}")
    
    # Experiment 3: Beam Search Size (for generation)
    beam_sizes = [1, 3, 5]
    model = MarianMt(base_model_name)  # Use same model for all beam size tests
    print(f"\n{'='*50}\nExperiment: Beam Search Size\n{'='*50}")
    
    # Prepare a smaller evaluation set for quick testing
    eval_dataset = tokenized_val.select(range(100))
    
    for beam_size in beam_sizes:
        try:
            # Create generate function with specified beam size
            def generate_with_beam(batch):
                tokenizer = model.get_tokenizer()
                inputs = {k: v for k, v in batch.items() if k != "labels"}
                
                # Move to correct device
                inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
                
                # Generate with specified beam size
                outputs = model.get_model().generate(
                    **inputs, 
                    num_beams=beam_size,
                    max_length=128
                )
                
                # Decode outputs
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                return decoded
            
            # Generate translations with this beam size
            start_time = time.time()
            translations = []
            references = []
            
            for batch in eval_dataset.select(range(100)):
                source_text = batch["en"]
                target_text = batch["de"]
                
                # Tokenize and generate
                inputs = model.tokenize_str(source_text)
                
                # Move to correct device 
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate
                outputs = model.get_model().generate(
                    **inputs,
                    num_beams=beam_size,
                    max_length=128
                )
                
                # Decode
                translated = model.get_tokenizer().decode(outputs[0], skip_special_tokens=True)
                
                translations.append(translated)
                references.append(target_text)
            
            # Calculate BLEU
            bleu_score = sacrebleu.corpus_bleu(translations, [references]).score
            generation_time = time.time() - start_time
            
            # Save results
            result = {
                "experiment": "beam_size",
                "value": beam_size,
                "bleu": bleu_score,
                "generation_time": generation_time
            }
            results.append(result)
            
            print(f"Beam Size {beam_size} results:")
            print(f"  BLEU Score: {bleu_score:.2f}")
            print(f"  Generation Time: {generation_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error in experiment with beam size {beam_size}: {e}")
    
    # Save all results to file
    results_file = os.path.join(experiment_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create plots
    plot_results(results, experiment_dir)
    
    return results

def plot_results(results, output_dir):
    """
    Create plots visualizing the experiment results.
    
    Args:
        results: List of experiment result dictionaries
        output_dir: Directory where plots will be saved
    """
    ensure_dir(output_dir)
    
    # Extract results by experiment type
    lr_results = [r for r in results if r["experiment"] == "learning_rate"]
    bs_results = [r for r in results if r["experiment"] == "batch_size"]
    beam_results = [r for r in results if r["experiment"] == "beam_size"]
    
    # Plot learning rate results
    if lr_results:
        lr_df = pd.DataFrame(lr_results)
        
        plt.figure(figsize=(12, 6))
        
        # BLEU score plot
        plt.subplot(1, 2, 1)
        plt.plot(lr_df["value"], lr_df["bleu"], marker="o", linestyle="-", color="blue")
        plt.xlabel("Learning Rate")
        plt.ylabel("BLEU Score")
        plt.title("Learning Rate vs BLEU Score")
        plt.xscale("log")
        plt.grid(True)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(lr_df["value"], lr_df["eval_loss"], marker="o", linestyle="-", color="orange", label="Eval Loss")
        plt.plot(lr_df["value"], lr_df["train_loss"], marker="x", linestyle="--", color="red", label="Train Loss")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate vs Loss")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "learning_rate_results.png"))
        plt.close()
    
    # Plot batch size results
    if bs_results:
        bs_df = pd.DataFrame(bs_results)
        
        plt.figure(figsize=(12, 6))
        
        # BLEU score plot
        plt.subplot(1, 2, 1)
        plt.plot(bs_df["value"], bs_df["bleu"], marker="o", linestyle="-", color="blue")
        plt.xlabel("Batch Size")
        plt.ylabel("BLEU Score")
        plt.title("Batch Size vs BLEU Score")
        plt.grid(True)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(bs_df["value"], bs_df["eval_loss"], marker="o", linestyle="-", color="orange", label="Eval Loss")
        plt.plot(bs_df["value"], bs_df["train_loss"], marker="x", linestyle="--", color="red", label="Train Loss")
        plt.xlabel("Batch Size")
        plt.ylabel("Loss")
        plt.title("Batch Size vs Loss")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "batch_size_results.png"))
        plt.close()
    
    # Plot beam size results
    if beam_results:
        beam_df = pd.DataFrame(beam_results)
        
        plt.figure(figsize=(12, 6))
        
        # BLEU score plot
        plt.subplot(1, 2, 1)
        plt.plot(beam_df["value"], beam_df["bleu"], marker="o", linestyle="-", color="blue")
        plt.xlabel("Beam Size")
        plt.ylabel("BLEU Score")
        plt.title("Beam Size vs BLEU Score")
        plt.grid(True)
        
        # Generation time plot
        plt.subplot(1, 2, 2)
        plt.plot(beam_df["value"], beam_df["generation_time"], marker="o", linestyle="-", color="green")
        plt.xlabel("Beam Size")
        plt.ylabel("Generation Time (s)")
        plt.title("Beam Size vs Generation Time")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "beam_size_results.png"))
        plt.close()
    
    # Create summary plot with all experiments
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    if lr_results:
        lr_df = pd.DataFrame(lr_results)
        plt.plot(range(len(lr_df)), lr_df["bleu"], marker="o", linestyle="-", label="Learning Rate Experiments")
        plt.xticks(range(len(lr_df)), [f"lr={v}" for v in lr_df["value"]])
    
    plt.subplot(3, 1, 2)
    if bs_results:
        bs_df = pd.DataFrame(bs_results)
        plt.plot(range(len(bs_df)), bs_df["bleu"], marker="s", linestyle="-", label="Batch Size Experiments")
        plt.xticks(range(len(bs_df)), [f"bs={v}" for v in bs_df["value"]])
    
    plt.subplot(3, 1, 3)
    if beam_results:
        beam_df = pd.DataFrame(beam_results)
        plt.plot(range(len(beam_df)), beam_df["bleu"], marker="^", linestyle="-", label="Beam Size Experiments")
        plt.xticks(range(len(beam_df)), [f"beam={v}" for v in beam_df["value"]])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_experiments_summary.png"))
    plt.close()

def create_summary(results, output_dir):
    """
    Create a summary text file of the experiment results.
    
    Args:
        results: List of experiment result dictionaries
        output_dir: Directory where summary will be saved
    """
    summary_file = os.path.join(output_dir, "experiment_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("===== HYPERPARAMETER EXPERIMENT SUMMARY =====\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of experiments: {len(results)}\n\n")
        
        # Group by experiment type
        experiment_types = set(r["experiment"] for r in results)
        
        for exp_type in experiment_types:
            f.write(f"----- {exp_type.upper()} EXPERIMENTS -----\n")
            type_results = [r for r in results if r["experiment"] == exp_type]
            type_results.sort(key=lambda x: x["value"])
            
            for r in type_results:
                f.write(f"\n{exp_type} = {r['value']}:\n")
                f.write(f"  BLEU Score: {r['bleu']:.2f}\n")
                
                if "eval_loss" in r:
                    f.write(f"  Evaluation Loss: {r['eval_loss']:.4f}\n")
                
                if "train_loss" in r:
                    f.write(f"  Training Loss: {r['train_loss']:.4f}\n")
                
                if "training_time" in r:
                    f.write(f"  Training Time: {r['training_time']:.2f} seconds\n")
                
                if "generation_time" in r:
                    f.write(f"  Generation Time: {r['generation_time']:.2f} seconds\n")
            
            f.write("\n")
            
            # Find best value
            best_result = max(type_results, key=lambda x: x["bleu"])
            f.write(f"Best {exp_type} value: {best_result['value']} (BLEU: {best_result['bleu']:.2f})\n\n")
        
        # Overall best result
        best_overall = max(results, key=lambda x: x["bleu"])
        f.write(f"===== BEST OVERALL RESULT =====\n")
        f.write(f"Experiment: {best_overall['experiment']} = {best_overall['value']}\n")
        f.write(f"BLEU Score: {best_overall['bleu']:.2f}\n")
        
        f.write("\n===== END OF SUMMARY =====\n")

if __name__ == "__main__":
    try:
        print("Starting Hyperparameter Experiment...")
        
        # Ensure output directory exists
        ensure_dir(OUTPUT_DIR)
        
        # Run experiments
        results = run_hyperparameter_experiment()
        
        # Create experiment timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(OUTPUT_DIR, f"experiment_{timestamp}")
        
        # Create summary
        create_summary(results, experiment_dir)
        
        print(f"\nExperiment completed. Results saved to {experiment_dir}")
        print("\nBest results per experiment type:")
        
        # Group by experiment type
        experiment_types = set(r["experiment"] for r in results)
        
        for exp_type in experiment_types:
            type_results = [r for r in results if r["experiment"] == exp_type]
            best_result = max(type_results, key=lambda x: x["bleu"])
            print(f"  Best {exp_type}: {best_result['value']} (BLEU: {best_result['bleu']:.2f})")
        
    except Exception as e:
        print(f"Error in experiment: {e}")
        raise