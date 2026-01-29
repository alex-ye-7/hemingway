from value import *
from nn import * 

x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.88)

x1w1 = x1*w1 + b 
print(x1w1)

# x = [2.0, 3.0, -1.0]
# n = MLP(3, [4,4,1])
# print(n(x))