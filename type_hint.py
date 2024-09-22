from typing import Any, List, Dict
from torch.nn import Module
from torch.optim import Optimizer
import torch

SentTokens = List[List[str]]
Tensor = torch.Tensor
Model = Module
Optim = Optimizer
Json = Dict[str, Any]
