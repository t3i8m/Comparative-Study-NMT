import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class M2M100Model():
    def __init__(self, model_name="facebook/m2m100_1.2B", device=None):
        """
        Initialize the M2M100 model and tokenizer.
        
        Args:
            model_name (str): Name of the M2M100 model (default: "facebook/m2m100_1.2B").
            device (str): Device to run the model on (default: auto-detect CUDA or CPU).
        """
        # set up the device (using CUDA if available, else CPU)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Load model and tokenizer
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        
    def translate(self, text, src_lang, tgt_lang, max_length=128):
        """
        Translate text from source language to target language.
        
        Args:
            text (str): Input text to translate.
            src_lang (str): Source language code (e.g., "en").
            tgt_lang (str): Target language code (e.g., "de").
            max_length (int): Maximum length of the generated translation.
        
        Returns:
            str: Translated text.
        """
        # Set source language
        self.tokenizer.src_lang = src_lang
        
        # Encode input
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate translation
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
            max_length=max_length
        )
        
        # Decode output
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation
    
#test the model if the model is working
if __name__ == "__main__":
    model = M2M100Model()
    text = "Hello, how are you?"
    src_lang = "en"
    tgt_lang = "de"
    translation = model.translate(text, src_lang, tgt_lang)
    print(f"Source: {text}")
    print(f"Translation: {translation}")

    def get_model(self):
        """Return the underlying M2M100 model."""
        return self.model

    def get_tokenizer(self):
        """Return the M2M100 tokenizer."""
        return self.tokenizer

    def get_device(self):
        """Return the device the model is running on."""
        return self.device