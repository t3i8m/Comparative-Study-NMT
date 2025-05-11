import nltk
import evaluate
from nltk.translate.bleu_score import corpus_bleu

from transformers_models.M2M100.m2m100 import M2M100Model
from transformers_models.marian.marianMT import MarianMt
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def retrieve_data(filenames):
    english_cleaned = []
    german_cleaned = []

    with open(f"data/{filenames[0]}", mode="r", encoding="utf-8") as f_en, \
            open(f"data/{filenames[1]}", mode="r", encoding="utf-8") as f_de:
        for en, de in tqdm(zip(f_en.readlines(), f_de.readlines())):
            english_cleaned.append(en.strip())
            german_cleaned.append(de.strip())

    return english_cleaned, german_cleaned

def compute_chrf(preds, refs):
    return chrf.compute(predictions=preds, references=refs)['score']

def evaluate_model(model, test_sentences, reference_translations, src_lang=None, tgt_lang=None):
    """Evaluate model's BLEU and ChrF scores."""
    translations = []
    for sentence in test_sentences:
        if isinstance(model, MarianMt):
            translations.extend(model.translate_text(sentence))
        elif isinstance(model, M2M100Model):
            translations.append(model.translate(sentence, src_lang, tgt_lang))

    list_of_references = [[nltk.word_tokenize(ref.lower())] for ref in reference_translations]
    hypotheses = [nltk.word_tokenize(trans.lower()) for trans in translations]

    bleu_score = corpus_bleu(list_of_references, hypotheses) * 100

    chrf_score = compute_chrf(preds=translations, refs=reference_translations)

    return bleu_score, chrf_score, translations

def add_noise(sentence):
    """Simulate noise by reversing first word."""
    words = sentence.split()
    if words:
        words[0] = words[0][::-1]
    return ' '.join(words)

def generate_noisy_set(sentences):
    return [add_noise(s) for s in sentences]

def compare_models():
    marian_model = MarianMt("Helsinki-NLP/opus-mt-en-de")
    m2m100_model = M2M100Model("facebook/m2m100_1.2B")

    test_sentences, reference_translations = retrieve_data(["test.en.txt", "test.de.txt"])

    noisy_test_sentences = generate_noisy_set(test_sentences)

    results = {}

    # MarianMT - Clean
    bleu, chrf_score, _ = evaluate_model(marian_model, test_sentences, reference_translations)
    results["MarianMT (clean)"] = {"BLEU": bleu, "ChrF": chrf_score}

    # MarianMT - Noisy
    bleu, chrf_score, _ = evaluate_model(marian_model, noisy_test_sentences, reference_translations)
    results["MarianMT (noisy)"] = {"BLEU": bleu, "ChrF": chrf_score}

    # M2M100 - Clean
    bleu, chrf_score, _ = evaluate_model(m2m100_model, test_sentences, reference_translations, src_lang="en", tgt_lang="de")
    results["M2M100 (clean)"] = {"BLEU": bleu, "ChrF": chrf_score}

    # M2M100 - Noisy
    bleu, chrf_score, _ = evaluate_model(m2m100_model, noisy_test_sentences, reference_translations, src_lang="en", tgt_lang="de")
    results["M2M100 (noisy)"] = {"BLEU": bleu, "ChrF": chrf_score}

    # Print summary
    print("\n=== Evaluation Summary ===")
    for model, scores in results.items():
        print(f"{model}: BLEU = {scores['BLEU']:.2f}, ChrF = {scores['ChrF']:.2f}")

if __name__ == "__main__":
    nltk.download('punkt')
    chrf = evaluate.load("chrf")
    compare_models()