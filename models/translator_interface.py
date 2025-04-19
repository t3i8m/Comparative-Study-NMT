from abc import ABC, abstractmethod
from typing import Any

class TranslatorInterface(ABC):

    @abstractmethod
    def get_model(self)->Any:
        """getter for the model"""
        pass

    @abstractmethod
    def get_tokenizer(self)->Any:
        """getter for the tokenizer"""
        pass
    
    @abstractmethod
    def translate_text(self, text: str) -> (list):
        "translate the text"
        pass
