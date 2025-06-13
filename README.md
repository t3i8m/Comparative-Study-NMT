# A Comparative Study of Machine Translation Models

This project provides a comprehensive framework for comparing the performance of different neural machine translation (NMT) architectures. It includes implementations and evaluation pipelines for a traditional LSTM-based Seq2Seq model and two state-of-the-art Transformer models: MarianMT and M2M100.

The primary goal is to analyze translation accuracy, robustness to noisy input, and the impact of fine-tuning strategies on the WMT14 English-German dataset.

Paper - https://drive.google.com/file/d/14VzlbmfQoo-pIUjypOm7-MfrPxCfmzsh/view?usp=sharing

## 🌟 Key Features

*   **Model Implementations**:
    *   A from-scratch **Seq2Seq LSTM** model with attention, built using PyTorch.
    *   Integration of pre-trained **MarianMT** (bilingual) and **M2M100** (multilingual) models from Hugging Face.
*   **Data Handling**:
    *   Scripts for downloading and preparing the WMT14 dataset.
    *   Data cleaning and preprocessing pipelines.
    *   Domain segmentation using Latent Dirichlet Allocation (LDA).
*   **Comprehensive Evaluation**:
    *   An evaluation pipeline using standard metrics: **BLEU**, **METEOR**, **BERTScore**, and **COMET**.
    *   Analysis to determine the most semantically accurate metric for the task.
*   **Robustness Testing**:
    *   A framework for injecting controlled stochastic and deterministic noise into input text to test model resilience.
*   **Reproducibility**:
    *   A `requirements.txt` file lists all Python dependencies.

## 📂 Project Structure

The repository is organized into modules for data handling, model implementation, and experimentation.
```
Comparative-Study-NMT/
├── data/                       # Scripts for downloading the WMT14 dataset
│   ├── Downloader.py
│   └── ...
├── data_processing/            # Data cleaning, BPE tokenization, and analysis
│   ├── CleanData.ipynb
│   ├── bpe_joint.model
│   └── bpe_joint.vocab
├── experiments/                # All experimental scripts and notebooks
│   ├── accurate_metric.ipynb   # Analysis of evaluation metrics
│   ├── domain_segmentation.ipynb # Topic modeling notebook
│   ├── fine_tunning.ipynb      # Fine-tuning logic for Transformer models
│   ├── experiment3_hyperparameters.py # Hyperparameter tuning experiments
│   └── ...
├── models/                     # All model implementations and interfaces
│   ├── translator_interface.py # A common interface for all models
│   ├── seq-to-seq-test-model/  # LSTM model implementation
│   │   └── LSTM.ipynb
│   └── transformers_models/    # Transformer model implementations
│       ├── marian/
│       │   └── marianMT.py
│       └── M2M100/
│           └── m2m100.py
├── .gitignore
├── README.md                   # You are here!
└── requirements.txt            # Python package dependencies
```
## 🚀 Getting Started

Follow these steps to set up the project environment and run the experiments.

### Prerequisites

*   Python 3.8+
*   PyTorch
*   Docker (optional, for containerized setup)

### 1. Installation

**Using a Virtual Environment**

```bash
# 1. Clone the repository
git clone https://github.com/your-username/translator-nlp-project.git
cd translator-nlp-project

# 2. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install the required dependencies
pip install -r requirements.txt
```
## 🚀 Usage and Running Experiments

The project is structured around a series of steps, from data preparation to model evaluation. Most of the logic is contained within Jupyter Notebooks (`.ipynb`) and Python scripts (`.py`) in the `data_processing` and `experiments` directories.

*   **Data Preparation**:
    *   Run the scripts in the `data/` directory to download the WMT14 dataset.
    *   Use `data_processing/CleanData.ipynb` to preprocess the text data, especially for the LSTM model.

*   **Training the LSTM Model**:
    *   Open and run the cells in `models/seq-to-seq-test-model/LSTM.ipynb` to train the LSTM model from scratch.

*   **Fine-Tuning Transformer Models**:
    *   Use `experiments/fine_tunning.ipynb` to fine-tune the MarianMT and M2M100 models on the WMT14 dataset.

*   **Running Evaluations**:
    *   The `experiments/` directory contains various scripts for evaluating the models. For example, `experiments/blue_score.py` can be used to calculate BLEU scores for model outputs.
    *   Notebooks like `accurate_metric.ipynb` guide the process of comparing different evaluation metrics.

## 🤖 Models Overview

*   **LSTM Seq2Seq**: A traditional encoder-decoder architecture with a bidirectional LSTM encoder and attention mechanism. It serves as a baseline to compare against modern architectures.
*   **MarianMT**: A highly optimized Transformer model specialized for bilingual translation. The model used in this project is pre-trained specifically for the English-German language pair.
*   **M2M100**: A massive multilingual Transformer model capable of translating between 100 different languages without relying on English as a pivot language. It demonstrates the power of large-scale multilingual transfer learning.

## 📜 Dependencies

Key dependencies are listed in `requirements.txt`. Major libraries include:

*   `torch`
*   `transformers`
*   `sacrebleu`
*   `bert-score`
*   `unbabel-comet`
*   `nltk`
*   `pandas`
*   `scikit-learn`
*   `jupyter`


## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
