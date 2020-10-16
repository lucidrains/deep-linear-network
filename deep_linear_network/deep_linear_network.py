import torch
from torch import nn
from functools import reduce

def mm(x, y):
    return x @ y

class DeepLinear(nn.Module):
    def __init__(self, dim_in, *dims):
        super().__init__()
        dims = [dim_in, *dims]
        pairs = list(zip(dims[:-1], dims[1:]))
        weights = list(map(lambda d: nn.Parameter(torch.randn(d)), pairs))
        self.weights = nn.ParameterList(weights)
        self._cache = None

    def forward(self, x):
        if self.training:
            self._cache = None
            return reduce(mm, self.weights, x)

        if self._cache is not None:
            return x @ self._cache

        head, *tail = self.weights
        weight = reduce(mm, tail, head)
        self._cache = weight
        return x @ weight
