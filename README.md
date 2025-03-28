This is a **prototype** machine translation project built in Python.  
It currently includes **two implemented models** as a proof of concept:

1. **Seq2Seq (RNN-based)** — a classic encoder-decoder architecture.
2. **Transformer (MarianMT via Hugging Face)** — a modern pretrained Transformer model for translation.

The goal is to test and compare how different architectures perform on translation tasks.  
This version is not a full production system — it's focused on getting core components working first.

## ✅ What’s Done

- [x] Basic Seq2Seq model
- [x] MarianMT model from Hugging Face Transformers
- [x] Tokenization and training pipeline
- [x] Docker support for isolated environment (almost)

## 🚧 In Progress / Next Steps

- Add more models (e.g., M2M100, Transformer from scratch)
- Add evaluation metrics (BLEU, etc.)
- Train on larger datasets (WMT14)
- Add logging and monitoring

