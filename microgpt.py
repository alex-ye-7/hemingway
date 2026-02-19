import os
import math
import random
random.seed(42)

# Input dataset
if not os.path.exists('/data/names.txt'):
    raise FileNotFoundError() 

# names.txt is a list of strings seperate by \n
docs = [l.strip() for l in open('/data/names.txt').read().strip().split('\n') if l.strip] 
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
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd) 
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)