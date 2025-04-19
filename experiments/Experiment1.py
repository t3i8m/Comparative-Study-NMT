import torch

def calculate_accuracy(predictions, references):
    correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    return correct / len(references) * 100

def evaluate_seq2seq_accuracy(seq2seq_model, test_sentences, reference_translations, eng_vocab, german_vocab, device):
    seq2seq_model.eval()
    predictions = [
        translate_sentence(sentence, seq2seq_model, eng_vocab, german_vocab)
        for sentence in test_sentences
    ]
    return calculate_accuracy(predictions, reference_translations)

def evaluate_transformer_accuracy(transformer_model, tokenizer, test_sentences, reference_translations, device):
    transformer_model.eval()
    predictions = []
    for sentence in test_sentences:
        tokens = tokenizer([sentence], return_tensors="pt", padding=True, truncation=True).to(device)
        translated = transformer_model.generate(**tokens)
        predictions.append(tokenizer.decode(translated[0], skip_special_tokens=True))
    return calculate_accuracy(predictions, reference_translations)

def compare_accuracies(seq2seq_model, transformer_model, test_sentences, reference_translations, eng_vocab, german_vocab, tokenizer, device):
    seq2seq_acc = evaluate_seq2seq_accuracy(seq2seq_model, test_sentences, reference_translations, eng_vocab, german_vocab, device)
    transformer_acc = evaluate_transformer_accuracy(transformer_model, tokenizer, test_sentences, reference_translations, device)
    return seq2seq_acc, transformer_acc