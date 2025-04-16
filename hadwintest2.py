import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer, MarianConfig
from torch.optim import AdamW
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
from datetime import datetime
from datasets import load_dataset
import evaluate

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Check GPU availability and configuration
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA device available")
        print("Checking NVIDIA drivers...")
        # Try to call nvidia-smi directly
        import subprocess
        try:
            result = subprocess.run(["nvidia-smi"], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            print(result.stdout)
        except Exception as e:
            print(f"nvidia-smi execution error: {e}")
except Exception as e:
    print(f"Error during GPU check: {e}")

# Basic settings
print("Setting up basic parameters...")
SRC_LANG = "en"
TGT_LANG = "de"
MODEL_NAME = f"Helsinki-NLP/opus-mt-{SRC_LANG}-{TGT_LANG}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load official datasets according to the proposal
print("Loading official datasets...")

# Function to load and prepare WMT dataset
def load_wmt_dataset(split='train', max_samples=None):
    print(f"Loading WMT14 English-German {split} dataset...")
    try:
        # Load WMT14 dataset
        dataset = load_dataset('wmt14', 'de-en', split=split)
        
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        print(f"WMT14 {split} dataset loaded: {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"Error loading WMT14 dataset: {e}")
        # Fallback to a smaller dataset or sample data
        print("Falling back to smaller dataset...")
        return None

# Function to load and prepare FLORES dataset
def load_flores_dataset(src_lang='eng', tgt_lang='deu', split='dev', max_samples=None):
    print(f"Loading FLORES-101 {src_lang}-{tgt_lang} {split} dataset...")
    try:
        # Load FLORES-101 dataset
        dataset = load_dataset('facebook/flores', 'all', split=split)
        
        # Extract the required language pair
        translated_data = []
        for item in dataset:
            if src_lang in item['sentence'] and tgt_lang in item['sentence']:
                translated_data.append({
                    "translation": {
                        "en": item['sentence'][src_lang],
                        "de": item['sentence'][tgt_lang]
                    }
                })
        
        if max_samples and max_samples < len(translated_data):
            translated_data = translated_data[:max_samples]
        
        print(f"FLORES {split} dataset loaded: {len(translated_data)} samples")
        return translated_data
    except Exception as e:
        print(f"Error loading FLORES dataset: {e}")
        # Fallback to a smaller dataset or sample data
        print("Falling back to smaller dataset...")
        return None

# Load actual datasets
try:
    # WMT14 English-German dataset (high resource)
    high_resource_train = load_wmt_dataset('train', max_samples=5000)  # 5000 samples for high resource
    high_resource_valid = load_wmt_dataset('validation', max_samples=100)
    
    # Subset of WMT14 (low resource simulation)
    low_resource_train = load_wmt_dataset('train', max_samples=500)    # 500 samples for low resource
    low_resource_valid = load_wmt_dataset('validation', max_samples=50)
    
    # FLORES dataset for additional evaluation (optional)
    flores_valid = load_flores_dataset(max_samples=100)
    
    # Create noisy versions for robustness testing
    def add_noise(text, noise_level=0.1):
        """Add spelling errors as noise to text"""
        if noise_level <= 0:
            return text
        
        chars = list(text)
        num_changes = max(1, int(len(chars) * noise_level))
        for _ in range(num_changes):
            idx = random.randint(0, len(chars) - 1)
            if random.random() < 0.33:  # Delete
                if len(chars) > 1:  # Ensure we don't delete all characters
                    chars.pop(idx)
            elif random.random() < 0.5:  # Replace
                chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
            else:  # Insert
                chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz'))
        return ''.join(chars)
    
    # Create noisy datasets
    def create_noisy_version(dataset, noise_level=0.1):
        """Create a noisy version of the dataset"""
        noisy_data = []
        for item in dataset:
            # Handle different dataset formats
            if 'translation' in item:
                src_text = item['translation']['en']
                tgt_text = item['translation']['de']
            else:
                # Adapt as needed for your dataset format
                src_text = item['en'] if 'en' in item else item['source']
                tgt_text = item['de'] if 'de' in item else item['target']
            
            noisy_src = add_noise(src_text, noise_level)
            noisy_data.append({
                'translation': {
                    'en': noisy_src,
                    'de': tgt_text
                }
            })
        return noisy_data
    
    # Create noisy versions if datasets were loaded successfully
    if high_resource_valid:
        noise_high_valid = create_noisy_version(high_resource_valid, 0.1)
    if low_resource_valid:
        noise_low_valid = create_noisy_version(low_resource_valid, 0.1)
    
    print("All datasets loaded successfully.")
except Exception as e:
    print(f"Error during dataset loading: {e}")
    print("Falling back to sample data...")
    
    # Fallback to sample data if official datasets cannot be loaded
    print("Creating sample datasets...")
    sample_data = [
        {"translation": {"en": "Hello, how are you?", "de": "Hallo, wie geht es dir?"}},
        {"translation": {"en": "I love machine translation.", "de": "Ich liebe maschinelle Übersetzung."}},
        {"translation": {"en": "The weather is nice today.", "de": "Das Wetter ist heute schön."}},
        {"translation": {"en": "This is a test sentence.", "de": "Dies ist ein Testsatz."}},
        {"translation": {"en": "What time is it?", "de": "Wie spät ist es?"}},
        {"translation": {"en": "Where is the nearest restaurant?", "de": "Wo ist das nächste Restaurant?"}},
        {"translation": {"en": "Can you help me, please?", "de": "Kannst du mir bitte helfen?"}},
        {"translation": {"en": "I need to buy a ticket.", "de": "Ich muss ein Ticket kaufen."}},
        {"translation": {"en": "The book is on the table.", "de": "Das Buch liegt auf dem Tisch."}},
        {"translation": {"en": "She walks to school every day.", "de": "Sie geht jeden Tag zur Schule."}},
        {"translation": {"en": "What is your name?", "de": "Wie heißt du?"}},
        {"translation": {"en": "I'm from the United States.", "de": "Ich komme aus den Vereinigten Staaten."}},
        {"translation": {"en": "Do you speak English?", "de": "Sprichst du Englisch?"}},
        {"translation": {"en": "How much does this cost?", "de": "Wie viel kostet das?"}},
        {"translation": {"en": "I'd like to order a coffee.", "de": "Ich möchte einen Kaffee bestellen."}},
        {"translation": {"en": "Where is the bathroom?", "de": "Wo ist die Toilette?"}},
        {"translation": {"en": "What do you do for a living?", "de": "Was machst du beruflich?"}},
        {"translation": {"en": "I enjoy reading books.", "de": "Ich lese gerne Bücher."}},
        {"translation": {"en": "The movie starts at 8 PM.", "de": "Der Film beginnt um 20 Uhr."}},
        {"translation": {"en": "How long have you been here?", "de": "Wie lange bist du schon hier?"}}
    ]
    
    # Duplicate sample data to create larger datasets
    high_resource_train = sample_data * 150  # 3000 samples
    high_resource_valid = sample_data[:10]   # 10 validation samples
    low_resource_train = sample_data * 15    # 300 samples 
    low_resource_valid = sample_data[:5]     # 5 validation samples
    
    # Create noisy versions for robustness testing
    noise_high_valid = create_noisy_version(high_resource_valid, 0.1)
    noise_low_valid = create_noisy_version(low_resource_valid, 0.1)

# Print dataset sizes
print(f"High-resource dataset: {len(high_resource_train)} training, {len(high_resource_valid)} validation")
print(f"Low-resource dataset: {len(low_resource_train)} training, {len(low_resource_valid)} validation")

# Load tokenizer
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

# Dataset class - updated to handle WMT and FLORES formats
class TranslationDataset(Dataset):
    def __init__(self, data, max_length=128):
        self.data = data
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different dataset formats
        if 'translation' in item:
            src = item['translation']['en']
            tgt = item['translation']['de']
        elif isinstance(item, dict) and 'en' in item and 'de' in item:
            src = item['en']
            tgt = item['de']
        else:
            # Try to extract text from other possible formats
            try:
                if hasattr(item, 'translation'):
                    src = item.translation.en
                    tgt = item.translation.de
                else:
                    # Default case, might need adjustments
                    src = str(item['source']) if 'source' in item else "Error: source not found"
                    tgt = str(item['target']) if 'target' in item else "Error: target not found"
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                src = "Error"
                tgt = "Fehler"
        
        # Use the recommended tokenization method
        inputs = tokenizer(
            src,
            text_target=tgt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": inputs.labels.squeeze()
        }

# Improved evaluation function
def evaluate(model, valid_data, device, metrics=None, verbose=False):
    model.eval()
    references = []
    hypotheses = []
    
    # Initialize metrics if provided
    if metrics is None:
        metrics = {}
    
    if verbose:
        print(f"Evaluating {len(valid_data)} samples...")
    
    for i, item in enumerate(tqdm(valid_data, desc="Evaluating", disable=not verbose)):
        # Extract source and reference text
        if 'translation' in item:
            src = item['translation']['en']
            tgt = item['translation']['de']
        elif isinstance(item, dict) and 'en' in item and 'de' in item:
            src = item['en']
            tgt = item['de']
        else:
            # Try to extract text from other possible formats
            try:
                if hasattr(item, 'translation'):
                    src = item.translation.en
                    tgt = item.translation.de
                else:
                    # Default case, might need adjustments
                    src = str(item['source']) if 'source' in item else "Error: source not found"
                    tgt = str(item['target']) if 'target' in item else "Error: target not found"
            except Exception as e:
                if verbose:
                    print(f"Error processing validation item {i}: {e}")
                continue
        
        # Translate source text
        inputs = tokenizer(src, return_tensors="pt", padding=True, truncation=True).to(device)
        try:
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=100)
            pred = tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            if verbose:
                print(f"Error generating translation for sample {i}: {e}")
            pred = ""
        
        if verbose and i < 5:  # Print first few samples
            print(f"Sample {i+1}:")
            print(f"  Source: '{src}'")
            print(f"  Reference: '{tgt}'")
            print(f"  Model output: '{pred}'")
        
        # Prepare for BLEU scoring
        references.append([tgt.split()])
        hypotheses.append(pred.split())
    
    # Calculate evaluation metrics
    results = {}
    
    # Calculate BLEU score
    if references and hypotheses:
        # Ensure at least one non-empty hypothesis
        non_empty = any(len(h) > 0 for h in hypotheses)
        if not non_empty:
            print("Warning: All model outputs are empty!")
            results["corpus_bleu"] = 0.0
            results["avg_sentence_bleu"] = 0.0
            results["accuracy"] = 0.0
        else:
            # Safely calculate BLEU
            try:
                corpus_bleu_score = corpus_bleu(references, hypotheses)
                results["corpus_bleu"] = corpus_bleu_score
            except Exception as e:
                print(f"Error calculating corpus BLEU: {e}")
                results["corpus_bleu"] = 0.0
            
            # Calculate sentence-level BLEU scores
            sentence_bleu_scores = []
            for ref, hyp in zip(references, hypotheses):
                try:
                    if len(hyp) > 0:  # Ensure non-empty hypothesis
                        score = sentence_bleu(ref, hyp)
                        sentence_bleu_scores.append(score)
                except Exception as e:
                    if verbose:
                        print(f"Error calculating sentence BLEU: {e}")
            
            # Calculate average sentence BLEU
            results["avg_sentence_bleu"] = sum(sentence_bleu_scores) / len(sentence_bleu_scores) if sentence_bleu_scores else 0.0
            
            # Calculate exact match accuracy
            exact_matches = sum(1 for ref, hyp in zip(references, hypotheses) 
                             if len(ref[0]) > 0 and ref[0] == hyp)
            results["accuracy"] = exact_matches / len(references) if references else 0.0
    else:
        print("Warning: No evaluation samples!")
        results["corpus_bleu"] = 0.0
        results["avg_sentence_bleu"] = 0.0
        results["accuracy"] = 0.0
    
    if verbose:
        print(f"Corpus BLEU Score: {results.get('corpus_bleu', 0.0):.4f}")
        print(f"Average Sentence BLEU: {results.get('avg_sentence_bleu', 0.0):.4f}")
        print(f"Accuracy: {results.get('accuracy', 0.0):.4f}")
    
    return results

# Training function with learning rate scheduling and gradient clipping
def train(model, train_data, learning_rate, batch_size, num_epochs=1, verbose=False):
    # Create data loader
    dataset = TranslationDataset(train_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer with learning rate scheduling for stability
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=verbose
    )
    
    # Training loop
    model.train()
    total_time = 0
    losses = []
    
    for epoch in range(num_epochs):
        if verbose:
            print(f"Starting training Epoch {epoch+1}/{num_epochs}...")
        start_time = time.time()
        total_loss = 0
        batch_losses = []
        
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}", disable=not verbose)):
            # Move data to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            batch_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Print loss every n batches
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}")
        
        # Calculate training time
        epoch_time = time.time() - start_time
        total_time += epoch_time
        
        # Calculate average loss
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        
        # Update learning rate based on loss
        scheduler.step(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch+1} completed, Average loss: {avg_loss:.4f}, Time: {epoch_time:.2f} seconds")
    
    # Add simple test checkpoint
    test_src = "Hello, how are you?"
    inputs = tokenizer(test_src, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    test_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Post-training test: '{test_src}' -> '{test_pred}'")
    
    return total_time, losses

# Create model with modified number of layers
def create_model_with_layers(num_encoder_layers, num_decoder_layers):
    try:
        # Get default configuration
        config = MarianConfig.from_pretrained(MODEL_NAME)
        
        # Modify number of layers
        original_encoder_layers = config.encoder_layers
        original_decoder_layers = config.decoder_layers
        
        print(f"Original model layers: Encoder={original_encoder_layers}, Decoder={original_decoder_layers}")
        print(f"Modifying to: Encoder={num_encoder_layers}, Decoder={num_decoder_layers}")
        
        config.encoder_layers = num_encoder_layers
        config.decoder_layers = num_decoder_layers
        
        # Create model with modified configuration
        model = MarianMTModel.from_pretrained(MODEL_NAME, config=config).to(DEVICE)
        
        # Verify layer count
        actual_encoder_layers = len(model.get_encoder().layers)
        actual_decoder_layers = len(model.get_decoder().layers)
        print(f"Actual model layers: Encoder={actual_encoder_layers}, Decoder={actual_decoder_layers}")
        
        return model
    except Exception as e:
        print(f"Error creating custom layer model: {e}")
        print("Falling back to default model")
        return MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

# Create experiment results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"experiment_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Function to plot results
def plot_results(results_df, output_path, metric='corpus_bleu', title=None):
    # Create high-resource and low-resource subsets
    high_df = results_df[results_df['Resource Type'] == 'High Resource']
    low_df = results_df[results_df['Resource Type'] == 'Low Resource']
    
    # Return if data is empty
    if high_df.empty and low_df.empty:
        print(f"Warning: No data available for plotting {metric}")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title or f'Effect of Hyperparameters on {metric}', fontsize=16)
    
    # Convert layer numbers to string to avoid pivot table errors
    if 'Num Layers' in results_df.columns:
        results_df['Num Layers'] = results_df['Num Layers'].astype(str)
    
    # High-resource heatmap
    if not high_df.empty:
        try:
            # Check for uniqueness of values before using pivot_table
            is_unique = high_df.groupby(['Learning Rate', 'Batch Size', 'Num Layers'])[metric].count().max() == 1
            
            if is_unique:
                pivot_high = high_df.pivot_table(
                    index='Learning Rate', 
                    columns=['Batch Size', 'Num Layers'], 
                    values=metric
                )
                sns.heatmap(pivot_high, annot=True, cmap="YlGnBu", ax=axes[0], fmt='.4f')
                axes[0].set_title('High Resource - ' + metric)
            else:
                print("Warning: High resource pivot table has duplicate values, using bar plot instead")
                sns.barplot(x='Learning Rate', y=metric, hue='Batch Size', data=high_df, ax=axes[0])
                axes[0].set_title('High Resource - ' + metric)
        except Exception as e:
            print(f"Error plotting high-resource heatmap: {e}")
            # Use scatter plot as fallback
            sns.scatterplot(x='Learning Rate', y=metric, hue='Batch Size', data=high_df, ax=axes[0])
            axes[0].set_title('High Resource - ' + metric)
    else:
        axes[0].text(0.5, 0.5, 'No High Resource Data', horizontalalignment='center', verticalalignment='center')
        axes[0].set_title('High Resource - ' + metric)
    
    # Low-resource heatmap
    if not low_df.empty:
        try:
            # Check for uniqueness of values before using pivot_table
            is_unique = low_df.groupby(['Learning Rate', 'Batch Size', 'Num Layers'])[metric].count().max() == 1
            
            if is_unique:
                pivot_low = low_df.pivot_table(
                    index='Learning Rate', 
                    columns=['Batch Size', 'Num Layers'], 
                    values=metric
                )
                sns.heatmap(pivot_low, annot=True, cmap="YlGnBu", ax=axes[1], fmt='.4f')
                axes[1].set_title('Low Resource - ' + metric)
            else:
                print("Warning: Low resource pivot table has duplicate values, using bar plot instead")
                sns.barplot(x='Learning Rate', y=metric, hue='Batch Size', data=low_df, ax=axes[1])
                axes[1].set_title('Low Resource - ' + metric)
        except Exception as e:
            print(f"Error plotting low-resource heatmap: {e}")
            # Use scatter plot as fallback
            sns.scatterplot(x='Learning Rate', y=metric, hue='Batch Size', data=low_df, ax=axes[1])
            axes[1].set_title('Low Resource - ' + metric)
    else:
        axes[1].text(0.5, 0.5, 'No Low Resource Data', horizontalalignment='center', verticalalignment='center')
        axes[1].set_title('Low Resource - ' + metric)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Quick test function
def quick_test():
    print("Running quick test...")
    
    # Test tokenizer and model functionality
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # Simple translation test
    test_sentences = [
        "Hello, how are you?",
        "This is a test.",
        "I love machine translation."
    ]
    
    for src in test_sentences:
        inputs = tokenizer(src, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Source: '{src}'")
        print(f"Translation: '{pred}'")
        print("---")
    
    # Test BLEU calculation
    reference = ["This", "is", "a", "test"]
    hypothesis = ["This", "is", "test"]
    bleu = sentence_bleu([reference], hypothesis)
    print(f"BLEU Test: {bleu:.4f}")
    
    # Test dataset class
    if high_resource_train and len(high_resource_train) > 0:
        print("Testing dataset class with real data...")
        dataset = TranslationDataset(high_resource_train[:2])
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Dataset sample keys: {list(sample.keys())}")
            print(f"Input shape: {sample['input_ids'].shape}")
    
    return True

# Comprehensive hyperparameter experiment - modified to match proposal requirements
def comprehensive_experiment():
    # Hyperparameter grid - according to proposal requirements
    learning_rates = [0.001, 0.01, 0.1]  # As specified in the proposal
    batch_sizes = [8, 16, 32]  # Larger batch sizes
    layer_combinations = [
        (6, 6),   # Standard model
        (3, 3),   # Small model
        (12, 12)  # Large model - more extreme contrast
    ]
    
    all_results = []
    
    # Run experiments for high-resource and low-resource languages
    for resource_type, (train_data, valid_data, noise_valid) in [
        ("High Resource", (high_resource_train, high_resource_valid, noise_high_valid)),
        ("Low Resource", (low_resource_train, low_resource_valid, noise_low_valid))
    ]:
        print(f"\n\n======== Starting {resource_type} Experiments ========")
        
        for lr in learning_rates:
            for bs in batch_sizes:
                for encoder_layers, decoder_layers in layer_combinations:
                    num_layers = f"{encoder_layers}_{decoder_layers}"
                    print(f"\n===== Testing LR={lr}, BS={bs}, Layers={num_layers} =====")
                    
                    # Create model with specified number of layers
                    model = create_model_with_layers(encoder_layers, decoder_layers)
                    
                    # Train
                    print(f"Starting training ({resource_type}, LR={lr}, BS={bs}, Layers={num_layers})...")
                    train_time, losses = train(model, train_data, learning_rate=lr, batch_size=bs, verbose=False)
                    
                    # Evaluate on clean data
                    print(f"Evaluating on clean data...")
                    clean_results = evaluate(model, valid_data, DEVICE)
                    
                    # Evaluate on noisy data
                    print(f"Evaluating on noisy data...")
                    noise_results = evaluate(model, noise_valid, DEVICE)
                    
                    # Calculate noise robustness
                    noise_robustness = noise_results["corpus_bleu"] / clean_results["corpus_bleu"] if clean_results["corpus_bleu"] > 0 else 0
                    
                    # Record results
                    result = {
                        "Resource Type": resource_type,
                        "Learning Rate": lr,
                        "Batch Size": bs,
                        "Num Layers": num_layers,
                        "Clean BLEU": clean_results["corpus_bleu"],
                        "Noise BLEU": noise_results["corpus_bleu"],
                        "Clean Sentence BLEU": clean_results["avg_sentence_bleu"],
                        "Noise Sentence BLEU": noise_results["avg_sentence_bleu"],
                        "Clean Accuracy": clean_results["accuracy"],
                        "Noise Accuracy": noise_results["accuracy"],
                        "Noise Robustness": noise_robustness,
                        "Training Time": train_time,
                        "Final Loss": losses[-1] if losses else 0
                    }
                    
                    all_results.append(result)
                    
                    print(f"Experiment completed: {resource_type}, LR={lr}, BS={bs}, Layers={num_layers}")
                    print(f"Clean data BLEU: {clean_results['corpus_bleu']:.4f}, Noisy data BLEU: {noise_results['corpus_bleu']:.4f}")
                    print(f"Noise robustness: {noise_robustness:.4f}")
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_path = os.path.join(results_dir, "hyperparameter_experiment_results.csv")
    results_df.to_csv(results_path)
    print(f"\nRaw results saved to: {results_path}")
    
    # Create visualizations
    plot_results(results_df, os.path.join(results_dir, "bleu_scores_heatmap.png"), 'Clean BLEU')
    plot_results(results_df, os.path.join(results_dir, "noise_robustness_heatmap.png"), 'Noise Robustness')
    
    # Calculate hyperparameter impact for high-resource and low-resource languages
    high_resource = results_df[results_df['Resource Type'] == 'High Resource']
    low_resource = results_df[results_df['Resource Type'] == 'Low Resource']
    
    # Impact of each factor on high-resource languages
    high_lr_impact = high_resource.groupby('Learning Rate')['Clean BLEU'].mean()
    high_bs_impact = high_resource.groupby('Batch Size')['Clean BLEU'].mean()
    high_layers_impact = high_resource.groupby('Num Layers')['Clean BLEU'].mean()
    
    # Impact of each factor on low-resource languages
    low_lr_impact = low_resource.groupby('Learning Rate')['Clean BLEU'].mean()
    low_bs_impact = low_resource.groupby('Batch Size')['Clean BLEU'].mean()
    low_layers_impact = low_resource.groupby('Num Layers')['Clean BLEU'].mean()
    
    # Create hyperparameter impact analysis report
    with open(os.path.join(results_dir, "hyperparameter_impact_analysis.txt"), "w") as f:
        f.write("====== Hyperparameter Impact Analysis on Translation Quality ======\n\n")
        
        f.write("=== High Resource Languages ===\n")
        f.write("Learning Rate Impact:\n")
        for lr, impact in high_lr_impact.items():
            f.write(f"  Learning Rate {lr}: Average BLEU Score {impact:.4f}\n")
        
        f.write("\nBatch Size Impact:\n")
        for bs, impact in high_bs_impact.items():
            f.write(f"  Batch Size {bs}: Average BLEU Score {impact:.4f}\n")
        
        f.write("\nLayer Count Impact:\n")
        for layers, impact in high_layers_impact.items():
            f.write(f"  Layers {layers}: Average BLEU Score {impact:.4f}\n")
        
        f.write("\n\n=== Low Resource Languages ===\n")
        f.write("Learning Rate Impact:\n")
        for lr, impact in low_lr_impact.items():
            f.write(f"  Learning Rate {lr}: Average BLEU Score {impact:.4f}\n")
        
        f.write("\nBatch Size Impact:\n")
        for bs, impact in low_bs_impact.items():
            f.write(f"  Batch Size {bs}: Average BLEU Score {impact:.4f}\n")
        
        f.write("\nLayer Count Impact:\n")
        for layers, impact in low_layers_impact.items():
            f.write(f"  Layers {layers}: Average BLEU Score {impact:.4f}\n")
        
        # Find best combinations
        if not high_resource.empty:
            best_high = high_resource.loc[high_resource['Clean BLEU'].idxmax()]
            f.write("\n\n=== Best Hyperparameter Combinations ===\n")
            f.write(f"High Resource Best Combination: Learning Rate={best_high['Learning Rate']}, Batch Size={best_high['Batch Size']}, "
                   f"Layers={best_high['Num Layers']}, BLEU={best_high['Clean BLEU']:.4f}\n")
        
        if not low_resource.empty:
            best_low = low_resource.loc[low_resource['Clean BLEU'].idxmax()]
            if 'best_high' not in locals():
                f.write("\n\n=== Best Hyperparameter Combinations ===\n")
            f.write(f"Low Resource Best Combination: Learning Rate={best_low['Learning Rate']}, Batch Size={best_low['Batch Size']}, "
                   f"Layers={best_low['Num Layers']}, BLEU={best_low['Clean BLEU']:.4f}\n")
        
        # Hyperparameter importance summary
        f.write("\n\n=== Hyperparameter Importance Summary ===\n")
        
        # Calculate variance of each hyperparameter to assess importance
        if not high_resource.empty:
            high_lr_var = high_resource.groupby('Learning Rate')['Clean BLEU'].std().fillna(0)
            high_bs_var = high_resource.groupby('Batch Size')['Clean BLEU'].std().fillna(0)
            high_layers_var = high_resource.groupby('Num Layers')['Clean BLEU'].std().fillna(0)
            
            # Calculate high-resource hyperparameter importance
            high_params = [
                ("Learning Rate", high_lr_var.mean()),
                ("Batch Size", high_bs_var.mean()),
                ("Layer Count", high_layers_var.mean())
            ]
            high_params.sort(key=lambda x: x[1], reverse=True)
            
            f.write("High Resource Language Hyperparameter Importance Ranking:\n")
            for param, importance in high_params:
                f.write(f"  {param}: Impact Factor {importance:.6f}\n")
        
        if not low_resource.empty:
            low_lr_var = low_resource.groupby('Learning Rate')['Clean BLEU'].std().fillna(0)
            low_bs_var = low_resource.groupby('Batch Size')['Clean BLEU'].std().fillna(0)
            low_layers_var = low_resource.groupby('Num Layers')['Clean BLEU'].std().fillna(0)
            
            # Calculate low-resource hyperparameter importance
            low_params = [
                ("Learning Rate", low_lr_var.mean()),
                ("Batch Size", low_bs_var.mean()),
                ("Layer Count", low_layers_var.mean())
            ]
            low_params.sort(key=lambda x: x[1], reverse=True)
            
            f.write("\nLow Resource Language Hyperparameter Importance Ranking:\n")
            for param, importance in low_params:
                f.write(f"  {param}: Impact Factor {importance:.6f}\n")
        
        # High-resource vs Low-resource comparison
        if not high_resource.empty and not low_resource.empty:
            f.write("\n\n=== High-Resource vs Low-Resource Comparison ===\n")
            
            high_mean = high_resource['Clean BLEU'].mean()
            low_mean = low_resource['Clean BLEU'].mean()
            
            f.write(f"High Resource Average BLEU: {high_mean:.4f}\n")
            f.write(f"Low Resource Average BLEU: {low_mean:.4f}\n")
            f.write(f"Performance Gap: {high_mean - low_mean:.4f} ({(high_mean/low_mean - 1)*100:.2f}%)\n")
            
            # Noise robustness comparison
            high_robustness = high_resource['Noise Robustness'].mean()
            low_robustness = low_resource['Noise Robustness'].mean()
            
            f.write(f"\nHigh Resource Noise Robustness: {high_robustness:.4f}\n")
            f.write(f"Low Resource Noise Robustness: {low_robustness:.4f}\n")
            f.write(f"Robustness Gap: {high_robustness - low_robustness:.4f}\n")
    
    print(f"Analysis report saved to: {os.path.join(results_dir, 'hyperparameter_impact_analysis.txt')}")
    
    # Display summary of main results
    print("\n\n=================== Experiment Results Summary ===================")
    
    # Calculate and display best combinations
    if not high_resource.empty:
        best_high = high_resource.loc[high_resource['Clean BLEU'].idxmax()]
        print("\nBest Hyperparameter Combinations:")
        print(f"High Resource Best: LR={best_high['Learning Rate']}, BS={best_high['Batch Size']}, "
             f"Layers={best_high['Num Layers']}, BLEU={best_high['Clean BLEU']:.4f}")
    
    if not low_resource.empty:
        best_low = low_resource.loc[low_resource['Clean BLEU'].idxmax()]
        if 'best_high' not in locals():
            print("\nBest Hyperparameter Combinations:")
        print(f"Low Resource Best: LR={best_low['Learning Rate']}, BS={best_low['Batch Size']}, "
             f"Layers={best_low['Num Layers']}, BLEU={best_low['Clean BLEU']:.4f}")
    
    # Display hyperparameter importance
    if 'high_params' in locals() and 'low_params' in locals():
        print("\nHyperparameter Importance Ranking:")
        print("High Resource: " + " > ".join([p[0] for p in high_params]))
        print("Low Resource: " + " > ".join([p[0] for p in low_params]))
    
    return results_df

if __name__ == "__main__":
    print("Preparing experiment...")
    try:
        # Make sure required packages are installed
        try:
            import evaluate
        except ImportError:
            print("Installing additional required packages...")
            import subprocess
            subprocess.check_call(["pip", "install", "evaluate", "sacrebleu", "sacremoses"])
            import evaluate
        
        # Run quick test first
        print("\n============== Running Quick Test ==============")
        if quick_test():
            print("\nQuick test passed, starting full experiment...")
            results_df = comprehensive_experiment()
            print("\nExperiment completed! All results and analysis saved to folder: " + results_dir)
        else:
            print("\nQuick test failed, please check basic functionality")
    except Exception as e:
        import traceback
        print(f"Error during experiment: {e}")
        traceback.print_exc()