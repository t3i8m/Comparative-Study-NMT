import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import random
from torch.utils.tensorboard import SummaryWriter

# Tokenizers
spacy_ger = get_tokenizer("spacy", language="de_core_news_sm")
spacy_eng = get_tokenizer("spacy", language="en_core_web_sm")

# Vocabulary building
def yield_tokens(data_iter, tokenizer):
    for src, tgt in data_iter:
        yield tokenizer(src)
        yield tokenizer(tgt)

train_iter = Multi30k(split='train', language_pair=('de', 'en'))


german_vocab = build_vocab_from_iterator(yield_tokens(train_iter, spacy_ger), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
english_vocab = build_vocab_from_iterator(yield_tokens(train_iter, spacy_eng), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
german_vocab.set_default_index(german_vocab["<unk>"])
english_vocab.set_default_index(english_vocab["<unk>"])

# Custom transform functions
def vocab_transform(vocab, tokens):
    return [vocab[token] for token in tokens]

def to_tensor(indices, padding_value):
    return torch.tensor(indices, dtype=torch.long)

# Transforms
def german_transform(tokens):
    indices = vocab_transform(german_vocab, tokens)
    return to_tensor(indices, padding_value=german_vocab["<pad>"])

def english_transform(tokens):
    indices = vocab_transform(english_vocab, tokens)
    return to_tensor(indices, padding_value=english_vocab["<pad>"])

# DataLoader
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(german_transform([f"<sos>"] + spacy_ger(src_sample) + [f"<eos>"]))
        tgt_batch.append(english_transform([f"<sos>"] + spacy_eng(tgt_sample) + [f"<eos>"]))
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=german_vocab["<pad>"])
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=english_vocab["<pad>"])
    return src_batch, tgt_batch

train_iter = Multi30k(split='train', language_pair=('de', 'en'))
train_loader = DataLoader(list(train_iter), batch_size=64, collate_fn=collate_fn)

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

# Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs.squeeze(0))
        return predictions, hidden, cell

# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english_vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        hidden, cell = self.encoder(source)
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

# Checkpoint saving and loading
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size_encoder = len(german_vocab)
input_size_decoder = len(english_vocab)
output_size = len(english_vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# TensorBoard
writer = SummaryWriter(f"runs/loss_plot")
step = 0

# Model
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
model = Seq2Seq(encoder_net, decoder_net).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english_vocab["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Training loop
for epoch in range(num_epochs):
    print(f"[Epoch {epoch + 1} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.train()
    for batch_idx, (inp_data, target) in enumerate(train_loader):
        inp_data = inp_data.to(device)
        target = target.to(device)

        output = model(inp_data, target)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1