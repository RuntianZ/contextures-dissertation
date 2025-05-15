import torch
from torch import Tensor  


def clip_singular_values(X: Tensor, cap: float = 1.0) -> Tensor:
    U, S, Vh = torch.linalg.svd(X)
    S[S > cap] = cap 
    return U @ torch.diag(S) @ Vh 

