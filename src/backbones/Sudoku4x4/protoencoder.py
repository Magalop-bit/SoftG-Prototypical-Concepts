import random
from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

NUM_DIGITS  = 4          # digit classes: 1, 2, 3, 4

# ── 1. ENCODER ───────────────────────────────────────────────────────────────

def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ProtoEncoder(nn.Module):
    """
    Input : (B, 1, 28, 28)
    Output: (B, 64)
    """
    def __init__(self, x_dim: int = 1, hid_dim: int = 64, z_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)   # (B, z_dim)

# ── 2. AUGMENTATION ──────────────────────────────────────────────────────────

_AUGMENT = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10,
    ),
])


def augment_support_set(
    images: torch.Tensor,          # (N, 1, 28, 28)
    labels: torch.Tensor,          # (N,)  values in {1,2,3,4}
    device: torch.device,
    num_augmentations: int = 9,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_imgs = [images]
    all_lbls = [labels]
    for _ in range(num_augmentations):
        aug = []
        for img in images:
            pil = TF.to_pil_image(img.cpu())
            aug.append(TF.to_tensor(_AUGMENT(pil)))
        all_imgs.append(torch.stack(aug).to(device))
        all_lbls.append(labels)
    return torch.cat(all_imgs), torch.cat(all_lbls)


# ── 3. EPISODIC SAMPLER ───────────────────────────────────────────────────────

def split_support_query(
    images: torch.Tensor,
    labels: torch.Tensor,
    k_support: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s_idx, q_idx = [], []
    for c in torch.unique(labels):
        idxs = (labels == c).nonzero(as_tuple=True)[0]
        perm = idxs[torch.randperm(len(idxs))]
        s_idx.append(perm[:k_support])
        if len(perm) > k_support:
            q_idx.append(perm[k_support:])
    s_idx = torch.cat(s_idx)
    q_idx = torch.cat(q_idx) if q_idx else torch.tensor([], dtype=torch.long)
    return images[s_idx], labels[s_idx], images[q_idx], labels[q_idx]


# ── 4. PROTOTYPICAL LOSS ──────────────────────────────────────────────────────

def prototypical_loss(
    embeddings: torch.Tensor,   # (N, D)
    labels: torch.Tensor,       # (N,)  — integer class ids (arbitrary values)
    n_support: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device  = embeddings.device
    classes = torch.unique(labels)

    support_proto, query_emb_list, query_lbl_list = [], [], []
    for i, c in enumerate(classes):
        idxs = (labels == c).nonzero(as_tuple=True)[0]
        support_proto.append(embeddings[idxs[:n_support]].mean(0))
        if len(idxs) > n_support:
            query_emb_list.append(embeddings[idxs[n_support:]])
            query_lbl_list.append(
                torch.full((len(idxs) - n_support,), i, dtype=torch.long, device=device)
            )

    if not query_emb_list:
        return torch.tensor(0.0, requires_grad=True, device=device), torch.tensor(1.0)

    prototypes = torch.stack(support_proto)
    query_emb  = torch.cat(query_emb_list)
    query_lbl  = torch.cat(query_lbl_list)

    dists   = torch.cdist(query_emb, prototypes, p=2) ** 2
    log_p_y = F.log_softmax(-dists, dim=1)
    loss    = F.nll_loss(log_p_y, query_lbl)
    acc     = (log_p_y.argmax(1) == query_lbl).float().mean()
    return loss, acc


# ── 5. PROTOTYPICAL HEAD ──────────────────────────────────────────────────────

def compute_prototypes(
    support_imgs: torch.Tensor,   # (N_s, 1, 28, 28)
    support_lbls: torch.Tensor,   # (N_s,)  values in {1,2,3,4}
    encoder: ProtoEncoder,
) -> torch.Tensor:
    """
    Returns prototypes of shape (4, D), one row per digit class 1-4.
    Row 0 → digit 1, row 1 → digit 2, …, row 3 → digit 4.
    Gradients flow through so Theorem 4.1 holds.
    """
    embs  = encoder(support_imgs)    # (N_s, D)
    D     = embs.size(1)
    device = embs.device
    proto = torch.zeros(NUM_DIGITS, D, device=device)
    for idx, d in enumerate(range(1, NUM_DIGITS + 1)):   # d ∈ {1,2,3,4}
        mask = support_lbls == d
        if mask.any():
            proto[idx] = embs[mask].mean(0)
    return proto                     # (4, D)


def digit_probs(
    embs: torch.Tensor,             # (B, D)  — already embedded
    prototypes: torch.Tensor,       # (4, D)
) -> torch.Tensor:
    """Returns (B, 4) — P(digit=1..4 | image)."""
    dists = torch.cdist(embs, prototypes, p=2) ** 2   # (B, 4)
    return F.softmax(-dists, dim=1)
