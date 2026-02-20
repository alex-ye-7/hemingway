import os
import math
import random
random.seed(42)

# Input dataset
if not os.path.exists('./data/names.txt'):
    raise FileNotFoundError() 

# names.txt is a list of strings seperate by \n
docs = [l.strip() for l in open('./data/names.txt').read().strip().split('\n') if l.strip] 
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer
uchars = sorted(set(''.join(docs))) # unique chars in dataset
BOS = len(uchars) # Beginning of sequence token
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# Autograd recursively applies chain rule through a computation graph
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data # scalar value of node, calculated during forward pass
        self.grad = 0 # derivative of loss w.r.t this node, calculated during backward pass
        self._children = children # how this node was created, or its children in the computation grpah
        self._local_grads = local_grads # local derivative of node w.r.t children, to be used for chain rule

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1,1))
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    # use local grads 
    def backward(self):
        visited = set()
        topo = []
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self) 
        self.grad = 1.0
        for v in reversed(topo): # start from the back
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad  # so clean

# Parameters
n_layer = 1 # how many layers of NN
n_embd = 16 # embedding dimenion
block_size = 16 # context length of attention window
n_head = 4 # number of attention heads
head_dim = n_embd // n_head # derived dim of each head
matrix = lambda nout, nin, std=0.08: [[(Value(random.gauss(0, std))) for _ in range(nin)] for _ in range(nout)]

# word token embedding, word position embedding, language model head -> from n_embd to respective outputs
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd) # output projection
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # list[Value]
print(f"num params: {len(params)}")

# Architecture 
def linear(x, w): # w @ x, Wx, nn.Linear(in, out)
    return [sum(wi*xi for wi,xi in zip(wo,x)) for wo in w] 

def softmax(logits): # numerical stability trick by using max value 
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits] # Value has .exp
    s = sum(exps)
    return [e / s for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5 # 1e-5 safety floor on denom 
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values): 
    tok_emb = state_dict['wte'][token_id] # 0..vocab_size-1
    pos_emb = state_dict['wpe'][pos_id] # 0...block_size-1 
    x = [t + p for t, p in zip(tok_emb, pos_emb)] 
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x) 
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = [] # (n_embd, n_embd)
        # attention
        for h in range(n_head):
            hs = h * head_dim # start
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))] # q @ k.T / sqrt(d_k)
            attn_weights = softmax(attn_logits)
             # matmul by hand -> weighted sum of each feature over time steps
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)) for j in range(head_dim))]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo']) 
        x = [a + b for a, b in zip(x, x_residual)] 
        # MLP
        x_residual = x 
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1']) 
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict[f'lm_head']) # (batch, n_embd) -> (batch, vocab_size)
    return logits 

lr, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8 # Adam optimizer parameters
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer 

num_steps = 1000 # training steps
for step in range(num_steps):
    # take a doc, tokenize, and insert BOS
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(char) for char in doc] + [BOS]
    n = min(block_size, len(tokens)-1)

    # build computation graph with forward pass + loss -> predict next character
    keys, values = [[] for _ in range(n_layer)],  [[] for _ in range(n_layer)] # (n_layer, n_embd)
    losses = []
    for pos_id in range(n): 
        token_id, target_id = tokens[pos_id], tokens[pos_id+1]
        logits = gpt(token_id, pos_id, keys, values) 
        probs = softmax(logits) # for every token 
        loss_t = -probs[target_id].log() # negative log likelihood or cross-entropy
        losses.append(loss_t)
    loss = (1/n) * sum(losses) 

    # Backward
    loss.backward()

    # Adam
    lr_t = lr * (1 - step / num_steps) # linear lr decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad # implicit m_i derived from m_(i-1)
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1)) # because step is zero indexed
        y_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr * m_hat / (y_hat ** 0.5 + eps_adam) # remember to push in negative direction
        p.grad = 0 # optimizer.zero_grad?

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:4f}", end='\r')

# Inference
temperature = 0.7 # range from 0 to 1
print("\n--- Running inference on model to create new, hallucinated names ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)],  [[] for _ in range(n_layer)]
    token_id = BOS 
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits]) # adding creativity
        token_id = random.choices(range(vocab_size), weights = [p.data for p in probs])[0] # defaults k=1 so need to extract
        if token_id == BOS:
            break 
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")