import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

def _row_predicate(vals: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Given a (B, 3) integer tensor of attribute values (shape OR color) for
    one row, return a dict of boolean tensors of shape (B,):
      same  – all three equal
      diff  – all three different
      two   – exactly two equal (i.e. not same, not diff)
    """
    a, b, c = vals[:, 0], vals[:, 1], vals[:, 2]
    same = (a == b) & (b == c)
    diff = (a != b) & (b != c) & (a != c)
    two  = ~same & ~diff
    return {"same": same, "diff": diff, "two": two}


def kand_label(concepts: torch.Tensor) -> torch.Tensor:
    """
    Compute the ground-truth Kand-Logic label for a batch of scenes.

    concepts : (B, N, N, 2)  where concepts[b, row, col, 0] = shape,
                                          concepts[b, row, col, 1] = color
    N = 3 (3×3 grid, but the rule is defined per-row across the 3 columns)

    Returns a (B,) bool tensor: True = positive (SAT) scene.

    K ⟺   (DiffS₁∧DiffS₂∧DiffS₃) ∨ (TwoS₁∧TwoS₂∧TwoS₃) ∨ (SameS₁∧SameS₂∧SameS₃)
         ∨ (DiffC₁∧DiffC₂∧DiffC₃) ∨ (TwoC₁∧TwoC₂∧TwoC₃) ∨ (SameC₁∧SameC₂∧SameC₃)
    """
    B = concepts.shape[0]

    # shapes/colors: (B, 3-rows, 3-cols)
    shapes = concepts[:, :, :, 0]   # (B, 3, 3)
    colors = concepts[:, :, :, 1]   # (B, 3, 3)

    # per-row predicates — each dict value is (B, 3) bool, one column per row
    sp = [_row_predicate(shapes[:, r, :]) for r in range(3)]   # shape predicates
    cp = [_row_predicate(colors[:, r, :]) for r in range(3)]   # color predicates

    # each clause requires the predicate to hold in ALL three rows
    diff_s = sp[0]["diff"] & sp[1]["diff"] & sp[2]["diff"]
    two_s  = sp[0]["two"]  & sp[1]["two"]  & sp[2]["two"]
    same_s = sp[0]["same"] & sp[1]["same"] & sp[2]["same"]

    diff_c = cp[0]["diff"] & cp[1]["diff"] & cp[2]["diff"]
    two_c  = cp[0]["two"]  & cp[1]["two"]  & cp[2]["two"]
    same_c = cp[0]["same"] & cp[1]["same"] & cp[2]["same"]

    return diff_s | two_s | same_s | diff_c | two_c | same_c


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
    labels = labels[:,0] * 3 + labels[:,1]

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
