import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PyPDF2 import PdfReader
import os
import re
import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import threading

# Hyperparameters & Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128  #512 for GPU
block_size = 128
max_iters = 3000
learning_rate = 3e-4
eval_iters = 50
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2
num_workers = 4 #10 for Ryzen 9
accumulation_steps = 4
model_dir = "saved_models"
model_filename = "gpt_language_model.pt"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, model_filename)

# Global variables for threading
train_loader = None
val_loader = None
vocab_size = None
encode = None
decode = None
data_loading_event = threading.Event()

def extract_text(filepath):
    if filepath.endswith(".txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    elif filepath.endswith(".pdf"):
        text = ""
        with open(filepath, 'rb') as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                text += page.extract_text()
    elif filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
        text = " ".join(df[df.columns[0]].astype(str).tolist())
    else:
        raise ValueError("Unsupported file type. Please provide a .txt, .pdf, or .parquet file.")
    return text

def clean_text(text, normalize=True):
    if normalize:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]  # Fix for invalid shape error
        return x, y

def load_and_preprocess_data(filepath, normalize=True, use_subword=True, num_workers=4):
    text = extract_text(filepath)
    text = clean_text(text, normalize)
    if use_subword:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
        tokenizer.train_from_iterator([text], trainer)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[("<s>", 0), ("</s>", 2)],
        )
        tokenizer.decoder = decoders.BPEDecoder()
        encoded_text = tokenizer.encode(text).ids
        vocab_size = tokenizer.get_vocab_size()
        encode = lambda s: tokenizer.encode(s).ids
        decode = lambda l: tokenizer.decode(l)
    else:
        chars = sorted(set(text))
        vocab_size = len(chars)
        string_to_int = {ch: i for i, ch in enumerate(chars)}
        int_to_string = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [string_to_int[c] for c in s]
        decode = lambda l: ''.join([int_to_string[i] for i in l])
        encoded_text = encode(text)

    data = torch.tensor(encoded_text, dtype=torch.long)
    train_dataset = TextDataset(data[:int(0.8 * len(data))], block_size)
    val_dataset = TextDataset(data[int(0.8 * len(data)):], block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, vocab_size, encode, decode

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self(index_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

def estimate_loss(model, val_loader, eval_iters):
    model.eval()
    losses = []
    with torch.no_grad():
        for iter, (xb, yb) in enumerate(val_loader):
            logits, loss = model.forward(xb.to(device), yb.to(device))
            losses.append(loss.item())
            if iter >= eval_iters:
                break
    model.train()
    return {'val': sum(losses) / len(losses)}

def train_model_in_process(rank, train_loader, val_loader, model, optimizer, scheduler, max_iters, eval_iters):
    model.train()
    total_tokens_processed = 0
    initial_tokens = 0  # Track initial tokens before training
    max_lr_reductions = 5
    lr_reductions = 0

    for iter in range(max_iters):
        print(f"Iteration {iter + 1}/{max_iters}")

        # Validation and Learning Rate Scheduling
        if iter % eval_iters == 0:
            losses = estimate_loss(model, val_loader, eval_iters)
            scheduler.step(losses['val'])
            print(f"Step: {iter}, Val loss: {losses['val']:.3f}")

            # Early Stopping Check
            if optimizer.param_groups[0]['lr'] < learning_rate / 10:
                lr_reductions += 1
                if lr_reductions >= max_lr_reductions:
                    print("Early stopping due to learning rate reduction.")
                    break

        # Training Loop (Batch Processing)
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (xb, yb) in enumerate(train_loader):
            # Track initial tokens for the first iteration
            if iter == 0 and batch_idx == 0:
                initial_tokens = xb.numel()

            logits, loss = model.forward(xb.to(device), yb.to(device))
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accumulation_steps
            total_tokens_processed += xb.numel()

            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Iteration {iter + 1}/{max_iters}, Batch {batch_idx + 1}/{num_batches}, Avg Loss: {avg_loss:.3f}, Tokens processed: {total_tokens_processed}")

        print(f"Iteration {iter + 1}/{max_iters} completed.")
        print(f"Total loss for iteration {iter + 1}: {total_loss:.3f}")

    # Save Model (Only on Rank 0)
    if rank == 0:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print(f"Initial tokens: {initial_tokens}")
    print(f"Total tokens processed during training: {total_tokens_processed}")
    # Calculate and display total tokens here (initial + added)
    total_tokens = initial_tokens + total_tokens_processed
    print(f"Total tokens: {total_tokens}") 


def prompt_model(model, encode, decode):
    model.eval()
    context = "The quick brown fox"
    context_encoded = torch.tensor([encode(context)], dtype=torch.long).to(device)
    output = model.generate(context_encoded, max_new_tokens=50)
    generated_text = decode(output[0].cpu().numpy().tolist())
    print(generated_text)

def main(data_file):
    print("Greetings from MyLLM - Your Ultimate LLM Companion!")
    global train_loader, val_loader, vocab_size, encode, decode

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"File '{data_file}' not found.")

    def data_loading_thread():
        try:
            global train_loader, val_loader, vocab_size, encode, decode
            train_loader, val_loader, vocab_size, encode, decode = load_and_preprocess_data(data_file, num_workers=num_workers)
        except Exception as e:
            print(f"Data loading thread error: {e}")

    def model_training_thread():
        try:
            model = GPTLanguageModel(vocab_size).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

            data_loading_event.wait()

            train_model_in_process(0, train_loader, val_loader, model, optimizer, scheduler, max_iters, eval_iters)
            prompt_model(model, encode, decode)
        except Exception as e:
            print(f"Model training thread error: {e}")

    data_thread = threading.Thread(target=data_loading_thread)
    model_thread = threading.Thread(target=model_training_thread)

    data_thread.start()
    data_thread.join()
    data_loading_event.set()
    model_thread.start()
    model_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT Language Model on text or PDF data.")
    parser.add_argument("data_file", type=str, help="Path to the text or PDF file for training.")
    args = parser.parse_args()

    main(args.data_file)
