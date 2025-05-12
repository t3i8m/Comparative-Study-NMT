import os
import random
import string
import sys
import nltk
import evaluate
from nltk.translate.bleu_score import corpus_bleu
from datasets import load_dataset

sys.path.append("C:/Users/Cecilia/Desktop/project 2-2/Translator_NLP_project")

from transformers_models.M2M100.m2m100 import M2M100Model
from transformers_models.marian.marianMT import MarianMt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def compute_chrf(preds, refs):
    return chrf.compute(predictions=preds, references=refs)['score']

def evaluate_model(model, test_sentences, reference_translations, src_lang=None, tgt_lang=None):
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

def add_noise(sentence, char_noise_prob=0.1, word_noise_prob=0.15):
    def random_char_edit(word):
        if not word:
            return word
        op = random.choice(["insert", "delete", "substitute", "swap"])
        i = random.randint(0, len(word) - 1)
        if op == "insert":
            return word[:i] + random.choice(string.ascii_lowercase) + word[i:]
        elif op == "delete" and len(word) > 1:
            return word[:i] + word[i+1:]
        elif op == "substitute":
            return word[:i] + random.choice(string.ascii_lowercase) + word[i+1:]
        elif op == "swap" and len(word) > 1 and i < len(word) - 1:
            return word[:i] + word[i+1] + word[i] + word[i+2:]
        return word

    words = sentence.split()
    new_words = []
    for word in words:
        if random.random() < word_noise_prob:
            continue
        if random.random() < char_noise_prob:
            word = random_char_edit(word)
        new_words.append(word)

    return ' '.join(new_words)

def generate_noisy_set(sentences, char_noise_prob=0.1, word_noise_prob=0.15):
    return [add_noise(s, char_noise_prob, word_noise_prob) for s in sentences]

def compare_models():
    print("Loading models...")
    marian_model = MarianMt("Helsinki-NLP/opus-mt-en-de")
    m2m100_model = M2M100Model("facebook/m2m100_1.2B")

    print("Loading data...")
    dataset = load_dataset("wmt14", "de-en", split="test[:100]")  
    test_sentences = [example["translation"]["en"] for example in dataset]
    reference_translations = [example["translation"]["de"] for example in dataset]


    noisy_test_sentences = generate_noisy_set(test_sentences, char_noise_prob=0.3, word_noise_prob=0.4)

    results = {}

    print("\nEvaluating MarianMT on Clean Data...")
    bleu, chrf_score, translations = evaluate_model(marian_model, test_sentences, reference_translations)
    results["MarianMT (clean)"] = {"BLEU": bleu, "ChrF": chrf_score}
    print(f"MarianMT (clean): BLEU = {bleu:.2f}, ChrF = {chrf_score:.2f}")
    for i in range(3):  # print just first 3 for brevity
        print(f"\nOriginal:  {test_sentences[i]}")
        print(f"Reference: {reference_translations[i]}")
        print(f"Translation: {translations[i]}")

    print("\nEvaluating MarianMT on Noisy Data...")
    bleu, chrf_score, translations = evaluate_model(marian_model, noisy_test_sentences, reference_translations)
    results["MarianMT (noisy)"] = {"BLEU": bleu, "ChrF": chrf_score}
    print(f"MarianMT (noisy): BLEU = {bleu:.2f}, ChrF = {chrf_score:.2f}")
    for i in range(3):
        print(f"\nNoisy Input:  {noisy_test_sentences[i]}")
        print(f"Reference:    {reference_translations[i]}")
        print(f"Translation:  {translations[i]}")

    print("\nEvaluating M2M100 on Clean Data...")
    bleu, chrf_score, translations = evaluate_model(m2m100_model, test_sentences, reference_translations, src_lang="en", tgt_lang="de")
    results["M2M100 (clean)"] = {"BLEU": bleu, "ChrF": chrf_score}
    print(f"M2M100 (clean): BLEU = {bleu:.2f}, ChrF = {chrf_score:.2f}")
    for i in range(3):
        print(f"\nOriginal:  {test_sentences[i]}")
        print(f"Reference: {reference_translations[i]}")
        print(f"Translation: {translations[i]}")

    print("\nEvaluating M2M100 on Noisy Data...")
    bleu, chrf_score, translations = evaluate_model(m2m100_model, noisy_test_sentences, reference_translations, src_lang="en", tgt_lang="de")
    results["M2M100 (noisy)"] = {"BLEU": bleu, "ChrF": chrf_score}
    print(f"M2M100 (noisy): BLEU = {bleu:.2f}, ChrF = {chrf_score:.2f}")
    for i in range(3):
        print(f"\nNoisy Input:  {noisy_test_sentences[i]}")
        print(f"Reference:    {reference_translations[i]}")
        print(f"Translation:  {translations[i]}")

    print("\n=== Evaluation Summary ===")
    for model, scores in results.items():
        print(f"{model}: BLEU = {scores['BLEU']:.2f}, ChrF = {scores['ChrF']:.2f}")

if __name__ == "__main__":
    nltk.download('punkt')
    chrf = evaluate.load("chrf")
    compare_models()


