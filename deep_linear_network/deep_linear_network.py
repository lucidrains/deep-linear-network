import torch
from torch import nn
from functools import reduce

class DeepLinear(nn.Module):
    def __init__(self, dim_in, *dims):
        super().__init__()
        dims = [dim_in, *dims]
        pairs = list(zip(dims[:-1], dims[1:]))
        weights = list(map(lambda d: nn.Parameter(torch.randn(d)), pairs))
        self.weights = nn.ParameterList(weights)
        self._cache = None

    def forward(self, x):
        head, *tail = self.weights
        weight = reduce(lambda x, y: x @ y, tail, head)
        self._cache = None if self.training else weight
        return x @ weight
