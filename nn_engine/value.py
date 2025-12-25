import math
from graphviz import Digraph

class Value:
    def __init__(self, data, _children=(), _op = '', label=''):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 
        self.label = label

    def relu(self):
        out = Value(max(0, self.data), (self,),'relu')
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1 / (1 + math.exp(-self.data))
        out = Value(sig, (self,), 'sigmoid')

        def _backward():
            self.grad += sig * (1 - sig) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topology = []
        visited = set()

        def build_graph(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_graph(child)
                topology.append(node)

        build_graph(self)

        self.grad = 1.0
        for node in reversed(topology):
            node._backward()
    
    def graph(self):
        dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
        visited = set()

        def add_nodes(v):
            if v not in visited:
                visited.add(v)
                label = f"{v.label}\n{v._op}\n{v.data}"
                dot.node(str(v), label=label)
                for child in v._prev:
                    dot.edge(str(child), str(v))
                    add_nodes(child)

        add_nodes(self)
        return dot

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):    
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
