import random
from value import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
        
    def parameters(self):
        return []
    
class Neuron(Module):
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1))

    def __call__(self, x): 
        # w * x + b
        act = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        return act.relu()
    
    def __repr__(self):
        return f"ReLU Neuron ({len(self.weights)})"
    
class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
class MLP(Module):
    def __init__(self, nin, nouts): # nin is int, nouts is list from layer
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 