from tqdm import tqdm

from transformer_models.marian.marianMT import MarianMt

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
translations = [translated(n, model) for n in english_cleaned_to_translate]

print(translations)
