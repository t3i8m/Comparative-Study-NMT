import random
import torch
from tqdm import tqdm
import nltk
from transformers import MarianMTModel, MarianTokenizer
from nltk.translate.bleu_score import corpus_bleu
from jiwer import wer
from comet import download_model, load_from_checkpoint


comet_model = load_from_checkpoint(download_model("wmt20-comet-da"))
nltk.download('punkt')

# Function to add noise to text
def add_noise_to_text(text, noise_level=0.2):
    tokens = text.split()
    num_noisy = int(len(tokens) * noise_level)
    noisy_indices = random.sample(range(len(tokens)), num_noisy)

    for i in noisy_indices:
        if len(tokens[i]) > 1:  
            j = random.randint(0, len(tokens[i]) - 1)
            tokens[i] = tokens[i][:j] + random.choice("abcdefghijklmnopqrstuvwxyz") + tokens[i][j + 1:]
    return " ".join(tokens)

# Function to retrieve data from files
def retrieve_data(filenames):
    english_cleaned = []
    german_cleaned = []

    with open(f"data/{filenames[0]}", mode="r", encoding="utf-8") as f_en, \
         open(f"data/{filenames[1]}", mode="r", encoding="utf-8") as f_de:
        for en, de in tqdm(zip(f_en.readlines(), f_de.readlines())):
            english_cleaned.append(en[:-1])
            german_cleaned.append(de[:-1])

    return english_cleaned, german_cleaned

# Function to translate sentences using Transformer model
def translate_using_transformer(model, tokenizer, sentences, device):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = model.generate(**inputs)
    translations = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translations

# Evaluate BLEU score for clean and noisy data
def evaluate_transformer_bleu(model, tokenizer, test_data, device, noise_level=0.0):
    original_sentences, reference_translations = zip(*test_data)
    predictions = []

    # Add noise to sentences
    noisy_sentences = [add_noise_to_text(sentence, noise_level) for sentence in original_sentences]

    # Translate both clean and noisy sentences
    for sentence in noisy_sentences:
        pred = translate_using_transformer(model, tokenizer, [sentence], device)
        predictions.append(pred)

    # Prepare the references and hypotheses for BLEU score calculation
    list_of_references = []
    hypotheses = []

    for ref, trans in zip(reference_translations, predictions):
        ref_tokens = nltk.word_tokenize(ref.lower())
        trans_tokens = nltk.word_tokenize(trans.lower())

        list_of_references.append([ref_tokens])
        hypotheses.append(trans_tokens)

    # Calculate BLEU score
    bleu_score = round(corpus_bleu(list_of_references, hypotheses), 3)
    return bleu_score

# Calculate WER score
def evaluate_wer(reference_translations, predictions):
    wer_scores = []
    for ref, pred in zip(reference_translations, predictions):
        wer_scores.append(wer(ref, pred))  
    avg_wer = sum(wer_scores) / len(wer_scores)  
    return avg_wer

# Calculate COMET score
def evaluate_comet(reference_translations, predictions):
    comet_scores = []
    for ref, pred in zip(reference_translations, predictions):
        comet_score = comet_model.predict(ref, pred)  
        comet_scores.append(comet_score)
    avg_comet = sum(comet_scores) / len(comet_scores)  
    return avg_comet


if __name__ == "__main__":

    cleaned = retrieve_data(["test.en", "test.de"])
    english_cleaned_to_translate = cleaned[0]
    german_cleaned_reference = cleaned[1]

    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_data = list(zip(english_cleaned_to_translate, german_cleaned_reference))

    # Evaluate BLEU score on clean data 
    bleu_clean = evaluate_transformer_bleu(model, tokenizer, test_data, device, noise_level=0.0)
    print(f"BLEU score for clean data: {bleu_clean:.4f}")
    print(f"BLEU score for clean data (%): {bleu_clean * 100:.2f}%")

    # Evaluate BLEU score on noisy data 
    bleu_noisy = evaluate_transformer_bleu(model, tokenizer, test_data, device, noise_level=0.3)  # level of noise is 30%
    print(f"BLEU score for noisy data: {bleu_noisy:.4f}")
    print(f"BLEU score for noisy data (%): {bleu_noisy * 100:.2f}%")

    # Translate sentences for WER and COMET evaluations
    noisy_sentences = [add_noise_to_text(sentence, 0.3) for sentence in english_cleaned_to_translate]  
    predictions_noisy = [translate_using_transformer(model, tokenizer, [sentence], device) for sentence in noisy_sentences]
    
    # Calculate WER
    wer_score = evaluate_wer(german_cleaned_reference, predictions_noisy)
    print(f"WER score for noisy data: {wer_score:.4f}")

    # Calculate COMET score
    comet_score = evaluate_comet(german_cleaned_reference, predictions_noisy)
    print(f"COMET score for noisy data: {comet_score:.4f}") 