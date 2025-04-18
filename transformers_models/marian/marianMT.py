from transformers import MarianMTModel, MarianTokenizer

class MarianMt():

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
    


