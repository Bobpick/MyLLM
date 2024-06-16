#Possible improvement. README.md is based on this
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PyPDF2 import PdfReader
import os
import re
import argparse
from transformers import GPT2Config, GPT2Model, GPTNeoConfig, GPTNeoModel, T5Config, T5Model, AutoTokenizer, AutoModelForCausalLM
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import threading
from torch.cuda.amp import autocast, GradScaler
import torch.profiler

# Hyperparameters & Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32 if device == torch.device('cpu') else torch.bfloat16
batch_size = 128
block_size = 128
max_iters = 3000
learning_rate = 3e-4
eval_iters = 50
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2
num_workers = 4
accumulation_steps = 4
model_dir = "saved_models"
model_filename = "language_model.pt"
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
    text = text.lower() if normalize else text

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.]', '', text)

    sentences = text.split(".")
    min_length = 5
    max_length = 50
    filtered_sentences = [s for s in sentences if min_length <= len(s.split()) <= max_length]
    text = ". ".join(filtered_sentences)

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
        y = chunk[1:]
        return x, y

def load_and_preprocess_data(filepath, normalize=True, use_subword=True, num_workers=4):
    text = extract_text(filepath)
    text = clean_text(text, normalize)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    encoded_text = tokenizer.encode(text)

    data = torch.tensor(encoded_text, dtype=torch.long)

    train_dataset = TextDataset(data[:int(0.8 * len(data))], block_size)
    val_dataset = TextDataset(data[int(0.8 * len(data)):], block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    vocab_size = tokenizer.vocab_size
    encode = tokenizer.encode
    decode = tokenizer.decode

    return train_loader, val_loader, vocab_size, encode, decode

class TransformerModel(nn.Module):
    def __init__(self, model_type, vocab_size):
        super().__init__()
        if model_type == "gpt2":
            config = GPT2Config(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
            self.model = GPT2Model(config)
        elif model_type == "gpt_neo":
            config = GPTNeoConfig(vocab_size=vocab_size, hidden_size=n_embd, num_layers=n_layer, num_heads=n_head)
            self.model = GPTNeoModel(config)
        elif model_type == "t5":
            config = T5Config(vocab_size=vocab_size, d_model=n_embd, num_layers=n_layer, num_heads=n_head)
            self.model = T5Model(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids, targets=None):
        outputs = self.model(input_ids)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            input_cond = input_ids[:, -block_size:]
            logits, _ = self(input_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids

def estimate_loss_and_perplexity(model, val_loader, eval_iters):
    model.eval()
    losses = []
    with torch.no_grad():
        for iter, (xb, yb) in enumerate(val_loader):
            logits, loss = model(xb.to(device), yb.to(device))
            losses.append(loss.item())
            if iter >= eval_iters:
                break
    model.train()
    avg_loss = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return {'val_loss': avg_loss, 'perplexity': perplexity.item()}

def train_one_epoch(model, train_loader, optimizer, scaler, accumulation_steps):
    model.train()
    total_loss = 0.0
    for batch_idx, (xb, yb) in enumerate(train_loader):
        with autocast():
            logits, loss = model(xb.to(device, dtype=dtype), yb.to(device, dtype=dtype))
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * accumulation_steps
    return total_loss / len(train_loader)

def train_model(rank, train_loader, val_loader, model, optimizer, scheduler, scaler, max_iters, eval_iters, model_type, encode, decode):
    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logdir'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    profiler.start()

    for iter in range(max_iters):
        print(f"Iteration {iter + 1}/{max_iters}")

        # Training loop
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, accumulation_steps)

        # Validation and perplexity calculation
        if iter % eval_iters == 0:
            metrics = estimate_loss_and_perplexity(model, val_loader, eval_iters)
            scheduler.step(metrics['val_loss'])
            print(f"Iteration {iter}, Val loss: {metrics['val_loss']:.3f}, Perplexity: {metrics['perplexity']:.3f}")

        # Generate text samples
        if iter % 100 == 0:
            prompt_model(model, encode, decode, model_type)

        profiler.step()

    profiler.stop()

    # Save model
    if rank == 0:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

def prompt_model(model, encode, decode, model_type):
    model.eval()
    context = "The quick brown fox"
    context_encoded = torch.tensor([encode(context)], dtype=torch.long).to(device)
    output = model.generate(context_encoded, max_new_tokens=50)
    generated_text = decode(output[0].cpu().numpy().tolist())
    print(f"Generated text after training: {generated_text}")

def main(data_file, model_type):
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
            model = TransformerModel(model_type, vocab_size).to(device, dtype=dtype)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
            scaler = GradScaler()

            data_loading_event.wait()

            train_model(0, train_loader, val_loader, model, optimizer, scheduler, scaler, max_iters, eval_iters, model_type, encode, decode)
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
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model on text or PDF data.")
    parser.add_argument("data_file", type=str, help="Path to the text or PDF file for training.")
    parser.add_argument("--model_type", type=str, choices=["gpt2", "gpt_neo", "t5"], default="gpt2", help="Type of Transformer model to use.")
    args = parser.parse_args()

    main(args.data_file, args.model_type)
