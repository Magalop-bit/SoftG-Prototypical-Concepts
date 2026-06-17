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
# 1.  ENCODER  (ProtoNet backbone – 4 conv-bn-relu-pool blocks, as in [Snell 2017]
#               and the r4rr repo backbones/addmnist_protonet.py)
# ---------------------------------------------------------------------------

def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ProtoEncoder(nn.Module):
    """
    4-block ConvNet → flat embedding.
    Input:  (B, 1, 28, 28)  [MNIST greyscale]
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
        return self.encoder(x).flatten(1)          # (B, z_dim)


# ---------------------------------------------------------------------------
# 2.  AUGMENTATION  (same transforms as the r4rr paper, Appendix E)
# ---------------------------------------------------------------------------

_AUGMENT = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10,
    ),
])


def augment_support_set(
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    num_augmentations: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (original + augmented) images and their repeated labels.

    images : (N, C, H, W)
    labels : (N,)
    Returns tensors on *device* of shape (N*(1+num_augmentations), C, H, W).
    """
    all_imgs = [images]
    all_lbls = [labels]

    for _ in range(num_augmentations):
        aug_batch = []
        for img in images:                         # img: (C, H, W)
            pil = TF.to_pil_image(img.cpu())
            aug_batch.append(TF.to_tensor(_AUGMENT(pil)))
        all_imgs.append(torch.stack(aug_batch).to(device))
        all_lbls.append(labels)

    return torch.cat(all_imgs, dim=0), torch.cat(all_lbls, dim=0)


# ---------------------------------------------------------------------------
# 3.  EPISODIC SAMPLER  (support / query split per class)
# ---------------------------------------------------------------------------

def split_support_query(
    images: torch.Tensor,
    labels: torch.Tensor,
    k_support: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each unique class sample k_support images as support; the rest go to
    the query set.  Returns (s_img, s_lbl, q_img, q_lbl).
    """
    s_idx, q_idx = [], []
    for c in torch.unique(labels):
        idxs = (labels == c).nonzero(as_tuple=True)[0]
        perm = idxs[torch.randperm(len(idxs))]
        s_idx.append(perm[:k_support])
        if len(perm) > k_support:
            q_idx.append(perm[k_support:])

    s_idx = torch.cat(s_idx)
    q_idx = torch.cat(q_idx) if q_idx else torch.tensor([], dtype=torch.long)

    return (
        images[s_idx], labels[s_idx],
        images[q_idx], labels[q_idx],
    )


# ---------------------------------------------------------------------------
# 4.  PROTOTYPICAL LOSS  (from Snell et al. 2017 / r4rr repo)
# ---------------------------------------------------------------------------

def prototypical_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_support: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard prototypical-networks episode loss.

    embeddings : (N, D)  – encoder output for support+query, interleaved by
                           the label ordering assumed by PrototypicalLoss
    labels     : (N,)    – integer class ids

    The function computes prototypes from the first n_support embeddings per
    class (in label order) and classifies the remaining ones.

    Returns (loss, accuracy).
    """
    device = embeddings.device
    classes = torch.unique(labels)
    n_classes = len(classes)

    # --- prototypes ---
    support_proto = []
    query_emb_list = []
    query_lbl_list = []

    for i, c in enumerate(classes):
        mask = labels == c
        idxs = mask.nonzero(as_tuple=True)[0]
        support_proto.append(embeddings[idxs[:n_support]].mean(0))
        if len(idxs) > n_support:
            query_emb_list.append(embeddings[idxs[n_support:]])
            query_lbl_list.append(
                torch.full((len(idxs) - n_support,), i, dtype=torch.long, device=device)
            )

    if not query_emb_list:
        # No query samples – return zero loss
        return torch.tensor(0.0, requires_grad=True), torch.tensor(1.0)

    prototypes = torch.stack(support_proto)        # (n_classes, D)
    query_emb  = torch.cat(query_emb_list, dim=0)  # (Q, D)
    query_lbl  = torch.cat(query_lbl_list, dim=0)  # (Q,)

    # --- squared Euclidean distances -> log-softmax -> NLL ---
    dists    = torch.cdist(query_emb, prototypes, p=2) ** 2  # (Q, n_classes)
    log_p_y  = F.log_softmax(-dists, dim=1)
    loss     = F.nll_loss(log_p_y, query_lbl)
    acc      = (log_p_y.argmax(1) == query_lbl).float().mean()

    return loss, acc


# ---------------------------------------------------------------------------
# 5.  PROTOTYPICAL HEAD  (computes P(digit | image) for a single image)
# ---------------------------------------------------------------------------

def compute_prototypes(
    support_imgs: torch.Tensor,
    support_lbls: torch.Tensor,
    encoder: ProtoEncoder,
) -> torch.Tensor:
    """
    Compute per-class centroids from (support_imgs, support_lbls).

    Returns prototypes of shape (10, D), one row per digit 0-9.
    The encoder is NOT called in no_grad here so gradients flow through it.
    """
    embs = encoder(support_imgs)                   # (N_s, D)
    D    = embs.size(1)
    device = embs.device
    proto = torch.zeros(10, D, device=device)

    for c in range(10):
        mask = support_lbls == c
        if mask.any():
            proto[c] = embs[mask].mean(0)
        # If class c is absent from the support set its prototype stays at 0;
        # in the full paper it is initialised with Gaussian noise (Eq. 5).
        # For simplicity and to keep this self-contained we use zeros.

    return proto                                   # (10, D)


def digit_probs(
    images: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """
    Given query images already embedded by the encoder and class prototypes,
    return softmax probabilities over 10 digit classes.

    images     : (B, D)  – already embedded
    prototypes : (10, D)

    Returns (B, 10).
    """
    dists = torch.cdist(images, prototypes, p=2) ** 2  # (B, 10)
    return F.softmax(-dists, dim=1)                    # (B, 10)
