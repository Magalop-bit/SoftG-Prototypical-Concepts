import math
import random
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 6.  SEMANTIC LOSS  for MNIST-EvenOdd
#
#  Background knowledge K:  Y = D1 + D2,  D1,D2 ∈ {0,...,9}
#
#  L_SL = -log P(Y = y)
#       = -log  Σ_{(d1,d2): d1+d2=y}  P(D1=d1) · P(D2=d2)
#
#  We precompute the (d1, d2) pairs for each possible sum y ∈ {0,...,18}.
# ---------------------------------------------------------------------------

# Build look-up: sum_value -> list of (d1, d2) pairs
_DIGIT_PAIRS: dict[int, list[tuple[int, int]]] = {s: [] for s in range(19)}
for d1, d2 in product(range(10), range(10)):
    _DIGIT_PAIRS[d1 + d2].append((d1, d2))


def semantic_loss_mnist(
    p1: torch.Tensor,
    p2: torch.Tensor,
    y:  torch.Tensor,
) -> torch.Tensor:
    """
    Semantic Loss for a batch of (image1, image2, sum_label) triples.

    p1 : (B, 10)  – P(D1 = d | image1)
    p2 : (B, 10)  – P(D2 = d | image2)
    y  : (B,)     – integer sum labels in {0,...,18}

    Returns scalar mean loss.
    """
    B = p1.size(0)
    device = p1.device
    log_prob_y = torch.zeros(B, device=device)

    # Accumulate P(Y = y) = Σ_{(d1,d2): d1+d2=y} p1[d1]*p2[d2]
    for sum_val, pairs in _DIGIT_PAIRS.items():
        # Boolean mask for samples whose label equals this sum
        mask = y == sum_val          # (B,)
        if not mask.any():
            continue
        # Σ_{(d1,d2)} p1[:,d1] * p2[:,d2]
        pair_prob = torch.zeros(B, device=device)
        for d1, d2 in pairs:
            pair_prob += p1[:, d1] * p2[:, d2]
        # Clamp for numerical safety before log
        log_prob_y[mask] = torch.log(pair_prob[mask].clamp(min=1e-12))

    return -log_prob_y.mean()