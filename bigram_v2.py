# Alexanader Ye 2025 

# Adding a head of self-attention to Bigram Language model

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 4 # how many indepedent sequences can you process in parallel?
block_size = 8 # how big of a context do you want to train on?
max_iters = 5000 # for training
learning_rate = 1e-3 # self-attention can't tolerate high lr
eval_interval = 500
eval_iters = 200
n_embd = 32 
n_heads = 4
dropout = 0.2

torch.manual_seed(1337)

with open("./data/snows.txt", "r", encoding='utf-8') as f:
    content = f.read()

# Create encoding functions for our model to understand
chars = sorted(list(set(content)))
vocab_size = len(chars)
char_to_i = {char : i for i, char in enumerate(chars)} # mapping from string to int
i_to_char = {i : char for i, char in enumerate(chars)} # reverse map
encode = lambda s: [char_to_i[char] for char in s]  # take a string, output int array
decode = lambda a: ''.join([i_to_char[i] for i in a]) # take an int array, output a string

# Train and test split
data = torch.tensor(encode(content), dtype=torch.long)
n = int(len(data) * 0.9) # first 90% will be training
train_data = data[:n]
val_data = data[n:]

# For training, setting up input x and target y's
def get_batch(split):
    """
    Output shapes will be (batch_size, block_size) or (B, T) 
    Think of x as 4 randomly sampled blocks in the text
    Think of y as the +1 offset of each sampled block
    So that target y at position (i,j) has its context in row/batch i in the first j columns/blocks
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size, (block_size,)) # sampling 4 blocks within text
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # goes with no_grad
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train() # turn back to training
    return out


class Head(nn.Module):
    """ head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    def forward(self, x): # input is (batch, time-step, channels or n_embd)
        B,T,C = x.shape
        k = self.key(x) # 
        q = self.query(x)
        # compute attention affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,16) @ (B,16,T) -> (B,T,T), normalized by 1/sqrt(d_k)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # prevents info from future tokens
        wei = F.softmax(wei, dim=-1) 
        # weighted aggregation of values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadSelfAttention(nn.Module):
    """
    multiple heads of attention in parallel 
    ensure num_heads * head_size == n_embd
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # initalize module list

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1) # concat along rows

class FeedForward(nn.Module):
    """ linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)        

# Adding attention mechanisms
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(n_embd) 
        self.sa_heads = MultiHeadSelfAttention(n_heads, n_embd // n_heads)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers that get_batch creates
        B,T = idx.shape
        tok_emd = self.token_embedding_table(idx) 
        pos_emd = self.position_embedding_table(torch.arange(T)) # (T,C) with 1,2,3...etc
        x = tok_emd + pos_emd # now has token identies AND positions at which tokens occur -> not useful until self-attention 
        x = self.sa_heads(x) # (B,T,C)
        x = self.ffwd(x) # (B,T,C)
        logits = self.lm_head(x) # logits becomes (B,T,vocab_size) 

        if targets is None:
            loss = None
        else:
            # Cross entropy loss takes channel as second argument in shape -> reshape
            # Torch view can make it desired shape
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens=1000):
        # Take a (B,T) and make it (B,T+1), (B,T+2), ..., (B, T+max)
        for _ in range(max_new_tokens):
            # b/c positional encoding, crop idx to latest block_size
            idx_last = idx[:, -block_size:]
            logits, loss = self(idx_last) # Pytorch trick equivalent to self.forward(idx)
            logits = logits[:, -1, :] # Focus on last token for each batch item, shape (B,C)
            probs = F.softmax(logits, dim=-1) # dim -1 means along the last dimension (vocab_size), so each batch is indepdent
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution
            idx = torch.cat((idx, idx_next), dim=1) # Along time dimension
        return idx

model = BigramLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
 
# Training loop
for iter in range(max_iters):
    # every once and while report losses
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from model
idx = torch.zeros((1,1), dtype=torch.long) # This starting point limits our batch to one sequence
output_data = decode(model.generate(idx, 400)[0].tolist()) # tolist to convert from tensor to list
print(output_data)

# with open("bigram_result.txt", "w") as file:
#     file.write(output_data)