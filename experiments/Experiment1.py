import nltk
import evaluate
from nltk.translate.bleu_score import corpus_bleu

from transformers_models.M2M100.m2m100 import M2M100Model
from transformers_models.marian.marianMT import MarianMt

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

chrf = evaluate.load("chrf")
meteor = evaluate.load("meteor")

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

    meteor_score = meteor.compute(predictions=translations, references=reference_translations)['meteor']

    return bleu_score, chrf_score, meteor_score, translations

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

    test_sentences = ["Hello, how are you?", "This is a test sentence."]
    reference_translations = ["Hallo, wie geht es dir?", "Dies ist ein Testsatz."]

    noisy_test_sentences = generate_noisy_set(test_sentences)

    results = {}

    # MarianMT - Clean
    bleu, chrf_score, meteor_score, _ = evaluate_model(marian_model, test_sentences, reference_translations)
    results["MarianMT (clean)"] = {"BLEU": bleu, "ChrF": chrf_score, "METEOR": meteor_score}

    # MarianMT - Noisy
    bleu, chrf_score, meteor_score, _ = evaluate_model(marian_model, noisy_test_sentences, reference_translations)
    results["MarianMT (noisy)"] = {"BLEU": bleu, "ChrF": chrf_score, "METEOR": meteor_score}

    # M2M100 - Clean
    bleu, chrf_score, meteor_score, _ = evaluate_model(m2m100_model, test_sentences, reference_translations, src_lang="en", tgt_lang="de")
    results["M2M100 (clean)"] = {"BLEU": bleu, "ChrF": chrf_score, "METEOR": meteor_score}

    # M2M100 - Noisy
    bleu, chrf_score, meteor_score, _ = evaluate_model(m2m100_model, noisy_test_sentences, reference_translations, src_lang="en", tgt_lang="de")
    results["M2M100 (noisy)"] = {"BLEU": bleu, "ChrF": chrf_score, "METEOR": meteor_score}

    print("\n=== Evaluation Summary ===")
    for model, scores in results.items():
        print(f"{model}: BLEU = {scores['BLEU']:.2f}, ChrF = {scores['ChrF']:.2f}, METEOR = {scores['METEOR']:.2f}")

if __name__ == "__main__":
    compare_models()