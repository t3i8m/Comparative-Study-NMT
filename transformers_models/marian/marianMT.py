from transformers import MarianMTModel, MarianTokenizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tqdm import tqdm
from models.translator_interface import TranslatorInterface

class MarianMt(TranslatorInterface):

    def __init__(self, model_name):
        # "Helsinki-NLP/opus-mt-en-de"
        self.model_name = model_name
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)

    def get_model(self)->(MarianMTModel):
        return self.model
    
    def get_tokenizer(self)->(MarianTokenizer):
        return self.tokenizer

    def translate_text(self, texts:str)->(list):
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = self.model.generate(**inputs)
        translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        return translations

    def batch_translate(self, texts:list, batch_size=32)->(list):
        translations = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.generate(**inputs)
            batch_translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(batch_translations)
            print(f"Batch {i // batch_size + 1} / {len(texts) // batch_size + 1}")
        return translations
    


