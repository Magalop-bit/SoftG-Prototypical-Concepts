from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ────────────────────────────────────────────────────────────────

NUM_DIGITS  = 4          # digit classes: 1, 2, 3, 4
BOARD_CELLS = 16         # 4 × 4

# All 24 permutations of {0,1,2,3} (indices into a 4-cell group).
_PERMS = list(permutations(range(NUM_DIGITS)))   # 24 tuples

# 4×4 Sudoku constraint groups (cell indices in a flattened 4×4 board):
#   4 rows, 4 columns, 4 2×2 boxes  → 12 groups of 4 cells each
def _sudoku_groups() -> list[list[int]]:
    groups = []
    # rows
    for r in range(4):
        groups.append([r*4 + c for c in range(4)])
    # columns
    for c in range(4):
        groups.append([r*4 + c for r in range(4)])
    # 2×2 boxes
    for br in range(2):
        for bc in range(2):
            groups.append([
                (br*2 + dr)*4 + (bc*2 + dc)
                for dr in range(2) for dc in range(2)
            ])
    return groups                                # 12 groups

_GROUPS = _sudoku_groups()


# ── 6. SEMANTIC LOSS FOR 4×4 SUDOKU ──────────────────────────────────────────

def _prob_group_is_permutation(
    group_probs: torch.Tensor,   # (B, 4, 4)  — p[b, cell_in_group, digit_idx]
) -> torch.Tensor:               # (B,)
    B = group_probs.size(0)
    device = group_probs.device
    total = torch.zeros(B, device=device)
    for perm in _PERMS:                               # 24 permutations
        prod = torch.ones(B, device=device)
        for i, j in enumerate(perm):
            prod = prod * group_probs[:, i, j]
        total = total + prod
    return total                                      # (B,)


def semantic_loss_sudoku(
    cell_probs: torch.Tensor,   # (B, 16, 4) — P(digit 1..4) for each of 16 cells
    y:          torch.Tensor,   # (B,)        — 0 or 1
) -> torch.Tensor:
    y = y.long().view(-1)

    B      = cell_probs.size(0)
    device = cell_probs.device

    log_p_valid = torch.zeros(B, device=device)
    for group in _GROUPS:                             # 12 groups of 4 cell indices
        gp = cell_probs[:, group, :]                  # (B, 4, 4)
        p_perm = _prob_group_is_permutation(gp)       # (B,)
        log_p_valid = log_p_valid + torch.log(p_perm.clamp(min=1e-12))

    p_valid   = torch.exp(log_p_valid.clamp(max=0.0))  # (B,)  in (0,1]
    p_invalid = (1.0 - p_valid).clamp(min=1e-12)       # (B,)

    loss_per_sample = torch.where(
        y == 1,
        -torch.log(p_valid.clamp(min=1e-12)),
        -torch.log(p_invalid),
    )
    return loss_per_sample.mean()