# Alexanader Ye 2025 

# Adding a decoding-only transformer to Bigram Language model

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 8 # how many indepedent sequences can you process in parallel?
block_size = 16 # how big of a context do you want to train on?
max_iters = 5000 # for training
learning_rate = 1e-3 # self-attention can't tolerate high lr
eval_interval = 500 # how often to output loss
eval_iters = 200 # for estimate_loss, how many tests to average against
n_embd = 64 # internal embedding dim
n_heads = 4 # how many heads of attention
n_layer = 4 # how many layers of network do you want
dropout = 0.05 # regularization

torch.manual_seed(1337)

with open("./data/The_Sun_Also_Rises.txt", "r", encoding='utf-8') as f:
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # input is (batch, time-step, channels or n_embd)
        B,T,C = x.shape
        k = self.key(x) # 
        q = self.query(x)
        # compute attention affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,16) @ (B,16,T) -> (B,T,T), normalized by 1/sqrt(d_k)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # prevents info from future tokens
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei) # randomly prevent some communication - prevents overfitting
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
        self.projection = nn.Linear(num_heads*head_size, n_embd) # linear transformation back into residual pathway
        self.dropout = nn.Dropout(dropout) # right before residual connection 

    def forward(self, x): # input is embeddings (B,T,C)
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concat along rows
        return self.projection(out)

# A combined `Head` and `MultiHeadAttention` into one class
# Processes heads in parallel, treating the heads as another batch dimension
class Attention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        self.key = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.query = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.value = nn.Linear(n_embd, num_heads * head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.projection = nn.Linear(num_heads*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        B,T,C = x.size() # C = num_heads*head_size

        # Project to all heads at once
        k = self.key(x) # (B, T, num_heads*head_size)
        q = self.query(x) # (B, T, num_heads*head_size)
        v = self.value(x) # (B, T, num_heads*head_size)

        # Reshape
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1,2) # (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1,2) # (B, num_heads, T, head_size)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1,2) # (B, num_heads, T, head_size)

        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5) # (B,nh,T,16) @ (B,nh,16,T) -> (B,nh,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # prevents info from future tokens
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei) 
        out = wei @ v # (B,nh,T,T) @ (B,nh,T,C) -> (B,nh,T,C)

        # Reshape back to (B,T,C)
        out = out.transpose(1,2).contiguous() # (B,T,nh,head_size)
        out = out.view(B,T, self.num_heads*self.head_size) 
        out = self.projection(out)
        return out


class FeedForward(nn.Module):
    """ linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Attention is All You Need paper has the inner layer as dimentionality 4x input/output embeddings
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout) # right before residual connection 
        )
    
    def forward(self, x): # input is embeddings (B,T,C)
        return self.net(x)        

class Block(nn.Module):
    """ 
    communication + computation
    n_embd: embedding dimension, n_heads: number of heads
    """
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        # self.sa = MultiHeadSelfAttention(n_heads, head_size) # communication
        self.sa = Attention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd) # computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x): # input is embeddings (B,T,C)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers that get_batch creates
        B,T = idx.shape
        tok_emd = self.token_embedding_table(idx) 
        pos_emd = self.position_embedding_table(torch.arange(T)) # (T,C) with 1,2,3...etc
        x = tok_emd + pos_emd # now has token identies AND positions at which tokens occur 
        x = self.blocks(x)
        x = self.ln(x)
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
# idx = torch.zeros((1,1), dtype=torch.long) # This starting point limits our batch to one sequence

# Generating with some context
context =  "The old man"
idx = torch.tensor([encode(context)], dtype=torch.long) 

output_data = decode(model.generate(idx, 400)[0].tolist()) # tolist to convert from tensor to list
print(output_data)

# with open("bigram_result.txt", "w") as file:
#     file.write(output_data)