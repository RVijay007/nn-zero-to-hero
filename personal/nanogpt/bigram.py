import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import datetime

# Hyperparameters for test run - ~210K parameters
batch_size = 16      # How many independent sequences will we process in parallel?
block_size = 32      # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
eval_iters = 200
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else device
n_embedding_dim = 64
n_heads = 4
n_layers = 4
dropout = 0.0

# Hyperparameters for full run - ~10M parameters
# batch_size = 64      # How many independent sequences will we process in parallel?
# block_size = 256      # What is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 500
# eval_iters = 200
# learning_rate = 3e-4
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else device
# n_embedding_dim = 384
# n_heads = 6
# n_layers = 6
# dropout = 0.2
# -------------------------------------------------

torch.manual_seed(1337)
start_time = time.time()

print(f'PyTorch version: {torch.__version__}')
print(f"Starting at {datetime.datetime.now()}")
print(f"Utilizing Device: {device}")

# Read input data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# encoder: take a string, output a list of integers
# decoder: take a list of integers, output a string
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Create datasets for training and validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(dataset_type):
    # generate a small batch of data of inputs x and targets y
    dataset = train_data if dataset_type == 'train' else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))

    x = torch.stack([dataset[i:i+block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for dataset in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(dataset)
            _, loss = model(X, Y)       # _ = logits, not used here
            losses[k] = loss.item()
        out[dataset] = losses.mean()    # Average losses across multiple batches to smooth loss curve
    model.train()

    return out

class Head(nn.Module):
    """One self-attention head"""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embedding_dim, head_size, bias=False)
        self.query = nn.Linear(n_embedding_dim, head_size, bias=False)
        self.value = nn.Linear(n_embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores, i.e. affinities
        wei = q @ k.transpose(-2,-1) * C**-0.5                          # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)                                    # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x)                                               # (B,T,C)
        out = wei @ v                                                   # (B, T, T) @ (B, T, C) -> (B, T, C)
        
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embedding_dim, n_embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),      # Growing network by 4
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),      # Shrinking network back down by 4
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding_dim)
        self.blocks = nn.Sequential(*[Block(n_embedding_dim, n_head=n_heads) for _ in range(n_layers)])
        self.layernorm_final = nn.LayerNorm(n_embedding_dim) # final layer norm
        self.language_model_head = nn.Linear(n_embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_embedding = self.token_embedding_table(idx)  # (B,T,Ce) - Batch, Time, Channels, i.e. (4,8,65)
        position_embedding = self.position_embedding_table(torch.arange(T, device=device))  # T,C

        x = token_embedding + position_embedding    # (B,T,C)
        x = self.blocks(x)                          # (B,T,C)
        x = self.layernorm_final(x)                 # (B,T,C)
        logits = self.language_model_head(x)        # (B,T,vocab_size)

        if targets is None:
            # Used in generation, not training
            loss = None
        else:
            # Pytorch cross_entropy requires Channel dimension be the second dimension,
            # so compress B,T into 1 dimension, B*T
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            
            # get the predictions
            logits, loss = self(idx_cond)
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx


model = BigramLanguageModel(vocab_size)
model.to(device)

# Create PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Print # of parameters
print('Model contains', sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# Training Loop
print(f'Running for {max_iters} iterations...')
for iter in range(max_iters):

    # Periodically, evaluate the loss on training and validation datasets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter} ({(time.time()-start_time):.3f}s): training loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Finished training at {datetime.datetime.now()}")

# Generate mew sequemce from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))

print(f"\nDone generating at {datetime.datetime.now()}")