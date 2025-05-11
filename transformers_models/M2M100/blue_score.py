from tqdm import tqdm
import sys
import os
import nltk
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers_models.marian.marianMT import MarianMt
nltk.download('punkt')  
from nltk.translate.bleu_score import corpus_bleu

#function to retrieve the data from the files
def retrieve_data(filenames):

    english_cleaned = []
    german_cleaned = []

    with open(f"data/{filenames[0]}", mode = "r", encoding = "utf-8") as f_en, \
        open(f"data/{filenames[1]}", mode = "r", encoding = "utf-8") as f_de:
        for en, de in tqdm(zip(f_en.readlines(), f_de.readlines())):
            english_cleaned.append(en[:-1])
            german_cleaned.append(de[:-1])

    return (english_cleaned, german_cleaned)

def translated(n, model):
    return model.translate_text(n)


cleaned = retrieve_data(["test.en", "test.de"])
english_cleaned_to_translate = cleaned[0]
german_cleaned_reference = cleaned[1]

model = MarianMt("Helsinki-NLP/opus-mt-en-de")
translations = model.batch_translate(english_cleaned_to_translate)

print(translations)
list_of_references = []
hypotheses = []

for ref, trans in zip(german_cleaned_reference, translations):

    ref_tokens = nltk.word_tokenize(ref.lower())
    trans_tokens = nltk.word_tokenize(trans.lower())

    list_of_references.append([ref_tokens])
    
    hypotheses.append(trans_tokens)

score = round(corpus_bleu(list_of_references, hypotheses), 3)
print("Corpus BLEU:", score)  
print("Corpus BLEU (%):", score * 100)