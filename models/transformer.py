import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer
from torch.optim import AdamW
from tqdm import tqdm


# sample implementation of the MarianMT transformer
SRC_LANG = "en"
TGT_LANG = "de"
MODEL_NAME = f"Helsinki-NLP/opus-mt-{SRC_LANG}-{TGT_LANG}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 5e-5

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

# sample
sample_data = [
    ("Hello, how are you?", "Hallo, wie geht es dir?"),
    ("This is a test sentence.", "Dies ist ein Testsatz."),
    ("The weather is nice today.", "Das Wetter ist heute schön."),
    ("I love machine translation.", "Ich liebe maschinelle Übersetzung.")
]


class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_tokenized = tokenizer.prepare_seq2seq_batch(
            src_texts=[src],
            tgt_texts=[tgt],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": src_tokenized.input_ids.squeeze(),
            "attention_mask": src_tokenized.attention_mask.squeeze(),
            "labels": src_tokenized.labels.squeeze()
        }

# DataLoader
train_loader = DataLoader(TranslationDataset(sample_data), batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LR)

model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

model.eval()
test_sentences = ["I like programming.", "Good morning!"]
tokens = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
translated = model.generate(**tokens)
results = tokenizer.batch_decode(translated, skip_special_tokens=True)
print("\n--- Translations ---")
for src, tgt in zip(test_sentences, results):
    print(f"{src} → {tgt}")
