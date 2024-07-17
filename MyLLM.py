import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from PyPDF2 import PdfReader
import pandas as pd
import os
import re
import argparse
from typing import Tuple, Callable
import logging
from tqdm import tqdm
import unicodedata
import faiss
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
class RAG:
    def __init__(self, documents, model_name='all-MiniLM-L6-v2'):
        self.documents = documents
        self.encoder = SentenceTransformer(model_name)
        self.index = self.create_index()

    def create_index(self):
        embeddings = self.encoder.encode(self.documents)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def retrieve(self, query, k=5):
        query_vector = self.encoder.encode([query])
        _, I = self.index.search(query_vector.astype('float32'), k)
        return [self.documents[i] for i in I[0]]
class ModelConfig:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.block_size = 128
        self.max_iters = 1000  # increased number of epochs
        self.learning_rate = 1e-3  # increased learning rate
        self.eval_interval = 50
        self.n_embd = 384
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.2
        self.num_workers = 0  # set to 0 to avoid multiprocessing issues
        self.model_dir = "saved_models"
        self.model_filename = "gpt_language_model.pt"
        self.log_dir = "logs"
        self.checkpoint_dir = "checkpoints"
        self.early_stopping_patience = 20
        self.target_loss = 0.05
        self.save_interval = 10  # Save interim model every 10 epochs
        self.interim_filename = "interim_model.pt"  # New parameter for interim model filename

# Create an instance of ModelConfig
config = ModelConfig()

# Set up logging
os.makedirs(config.log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(config.log_dir, 'training.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
def extract_text(filepath: str) -> str:
    if filepath.endswith(".txt"):
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode the file with any of the following encodings: {encodings}")
    elif filepath.endswith(".pdf"):
        with open(filepath, 'rb') as f:
            pdf = PdfReader(f)
            return " ".join(page.extract_text() for page in pdf.pages)
    elif filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
        return " ".join(df[df.columns[0]].astype(str).tolist())
    else:
        raise ValueError("Unsupported file type. Please provide a .txt, .pdf, or .parquet file.")

def clean_text(text: str, normalize: bool = True, handle_unicode: bool = True) -> str:
    if normalize:
        text = text.lower()
    if handle_unicode:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace characters with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove all non-word characters except spaces
    return text.strip()  # Remove leading and trailing whitespace

def create_tokenizer(text: str) -> Tokenizer:
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
    return tokenizer

def create_encoding_functions(use_subword: bool, text: str):
    if use_subword:
        tokenizer = create_tokenizer(text)
        encode = lambda s: tokenizer.encode(s).ids
        decode = lambda l: tokenizer.decode(l)
        vocab_size = tokenizer.get_vocab_size()
    else:
        chars = sorted(set(text))
        vocab_size = len(chars)
        string_to_int = {ch: i for i, ch in enumerate(chars)}
        int_to_string = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [string_to_int[c] for c in s]
        decode = lambda l: ''.join([int_to_string[i] for i in l])
    
    return encode, decode, vocab_size

class IterableTextDataset(IterableDataset):
    def __init__(self, filepath: str, block_size: int, encode_fn: Callable):
        self.filepath = filepath
        self.block_size = block_size
        self.encode_fn = encode_fn
        self.length = None

    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        encoded = self.encode_fn(text)
        self.length = len(encoded) - self.block_size
        for i in range(0, self.length):
            chunk = encoded[i:i + self.block_size + 1]
            x = chunk[:-1]
            y = chunk[1:]
            yield torch.tensor(x), torch.tensor(y)

    def __len__(self):
        if self.length is None:
            for _ in self:
                pass
        return self.length

def load_and_preprocess_data(filepath: str, config: ModelConfig, normalize: bool = True, use_subword: bool = True, handle_unicode: bool = True) -> Tuple[DataLoader, DataLoader, int, Callable, Callable]:
    text = extract_text(filepath)
    text = clean_text(text, normalize, handle_unicode)

    encode, decode, vocab_size = create_encoding_functions(use_subword, text)

    train_dataset = IterableTextDataset(filepath, config.block_size, encode)
    val_dataset = IterableTextDataset(filepath, config.block_size, encode)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

    return train_loader, val_loader, vocab_size, encode, decode

class Head(nn.Module):
    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
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
    def __init__(self, num_heads: int, head_size: int, n_embd: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, config.block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
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
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, config.dropout)
        self.ffwd = FeedForward(n_embd, config.dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig, rag: RAG, encode_fn: Callable, decode_fn: Callable):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        self.block_size = config.block_size
        self.rag = rag
        self.encode = encode_fn
        self.decode = decode_fn

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        generated = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            # RAG retrieval
            input_text = self.decode(idx_cond[0].tolist())
            retrieved_docs = self.rag.retrieve(input_text)
            context = " ".join(retrieved_docs) + " " + input_text

            # Encode the context back to token indices
            context_idx = torch.tensor(self.encode(context), dtype=torch.long).unsqueeze(0).to(idx.device)

            # Use the last self.block_size tokens if context is too long
            if context_idx.size(1) > self.block_size:
                context_idx = context_idx[:, -self.block_size:]

            logits, _ = self(context_idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            generated.append(idx_next.item())

        return idx, generated

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: ModelConfig):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    interim_path = os.path.join(config.checkpoint_dir, config.interim_filename)

    for epoch in range(config.max_iters):
        model.train()
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.max_iters}")
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': total_loss / num_batches, 'lr': optimizer.param_groups[0]['lr']})

        avg_train_loss = total_loss / num_batches
        val_loss = evaluate_model(model, val_loader, config)
        scheduler.step(val_loss)

        logging.info(
            f"Epoch {epoch + 1}: Train loss {avg_train_loss:.4f}, Val loss {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.model_dir, "best_" + config.model_filename))
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Save interim model
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, interim_path)
            logging.info(f"Interim model saved: {interim_path}")

        # Stop if target loss is reached
        if val_loss <= config.target_loss:
            logging.info(f"Target loss of {config.target_loss} reached after {epoch + 1} epochs. Stopping training.")
            break

    return model

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(config.device), y.to(config.device)
            logits, loss = model(x, y)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches

def save_model(model: nn.Module, filepath: str):
    torch.save({
        'model_state_dict': model.state_dict(),
        'rag_documents': model.rag.documents,
        'vocab_size': len(model.token_embedding_table.weight),
    }, filepath)

def load_model(filepath: str, model_class, vocab_size: int, config: ModelConfig, rag: RAG, encode_fn: Callable, decode_fn: Callable):
    checkpoint = torch.load(filepath)
    model = model_class(vocab_size, config, rag, encode_fn, decode_fn)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.rag.documents = checkpoint['rag_documents']
    model.rag.index = model.rag.create_index()  # Recreate the index
    return model


def main(args):
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Load and preprocess data
    train_loader, val_loader, vocab_size, encode, decode = load_and_preprocess_data(
        args.filepath, config, args.normalize, args.use_subword, args.handle_unicode
    )

    # Load documents for RAG (only once)
    with open(args.filepath, 'r', encoding='utf-8') as f:
        documents = f.read().split('\n')  # Split by paragraphs or sentences as needed
    rag = RAG(documents)

    if args.generate:
        model = load_model(os.path.join(config.model_dir, config.model_filename), GPTLanguageModel, vocab_size, config, rag, encode, decode)
        model = model.to(config.device)
        model.eval()

        context = torch.tensor(encode(args.start_text), dtype=torch.long).unsqueeze(0).to(config.device)
        generated_indices, _ = model.generate(context, args.max_new_tokens, args.temperature)
        generated_text = decode(generated_indices[0].tolist())
        print("Generated text:")
        print(generated_text)
        logging.info(f"Generated text: {generated_text}")
    else:
        interim_path = os.path.join(config.checkpoint_dir, config.interim_filename)
        if args.resume and os.path.exists(interim_path):
            # Load the interim model
            checkpoint = torch.load(interim_path)
            model = GPTLanguageModel(vocab_size, config, rag, encode, decode).to(config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resuming training from epoch {start_epoch}")
        else:
            model = GPTLanguageModel(vocab_size, config, rag, encode, decode).to(config.device)
            start_epoch = 0

        model = train_model(model, train_loader, val_loader, config)
        save_model(model, os.path.join(config.model_dir, config.model_filename))
        logging.info(f"Model saved to {os.path.join(config.model_dir, config.model_filename)}")
        print(f"Model saved to {os.path.join(config.model_dir, config.model_filename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Generate Text using a GPT model.")
    parser.add_argument("filepath", type=str, help="Path to the input text, PDF, or Parquet file.")
    parser.add_argument("--generate", action="store_true", help="Flag to indicate text generation instead of training.")
    parser.add_argument("--start_text", type=str, default="", help="Starting text for text generation.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for text generation.")
    parser.add_argument("--normalize", action="store_true", help="Normalize text during preprocessing.")
    parser.add_argument("--use_subword", action="store_true", help="Use subword tokenization.")
    parser.add_argument("--handle_unicode", action="store_true", help="Handle Unicode characters during preprocessing.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the interim checkpoint.")
    args = parser.parse_args()

    main(args)
