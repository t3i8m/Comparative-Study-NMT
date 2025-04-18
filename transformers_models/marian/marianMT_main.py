from tqdm import tqdm


english_cleaned = []
german_cleaned = []

with open("data/test.en", mode = "r", encoding = "utf-8") as f_en, \
    open("data/test.de", mode = "r", encoding = "utf-8") as f_de:
    for en, de in tqdm(zip(f_en.readlines(), f_de.readlines())):
        english_cleaned.append(en[:-1])
        german_cleaned.append(de[:-1])

print(english_cleaned)
print(german_cleaned)