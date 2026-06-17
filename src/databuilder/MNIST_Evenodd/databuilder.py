import torch
import torchvision
import numpy as np
import random

# ── Training pairs (same parity only) ──────────────────────────────────────
# 8 even+even pairs and 8 odd+odd pairs, matching Table 6.
# Sums: (0+6)=6, (2+8)=10, (4+6)=10, (4+8)=12  |  (1+5)=6, (3+7)=10, (1+9)=10, (3+9)=12
TRAIN_EVEN_PAIRS = [(0, 6), (2, 8), (4, 6), (4, 8),
                    (6, 0), (8, 2), (6, 4), (8, 4)]   # include both orderings → 8 pairs

TRAIN_ODD_PAIRS  = [(1, 5), (3, 7), (1, 9), (3, 9),
                    (5, 1), (7, 3), (9, 1), (9, 3)]   # include both orderings → 8 pairs

TRAIN_PAIRS = TRAIN_EVEN_PAIRS + TRAIN_ODD_PAIRS       # 16 pairs total


def _build_label_index(labels: torch.Tensor) -> dict:
    """Map each digit value → list of dataset indices that contain it."""
    index: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels.tolist()):
        index.setdefault(int(lbl), []).append(i)
    return index


def get_anchors(train_imgs, train_labels):
    """Return one canonical image per class (sorted by class label)."""
    anchor_label, anchor_image = [], []
    seen = set()
    for i in range(len(train_labels)):
        lbl = train_labels[i].item()
        if lbl not in seen:
            seen.add(lbl)
            anchor_label.append(train_labels[i])
            anchor_image.append(train_imgs[i])

    anchor_image = torch.stack(anchor_image)
    anchor_label = torch.stack(anchor_label)
    sort_idx = torch.argsort(anchor_label)
    return anchor_image[sort_idx]


def _sample_from_pairs(imgs, labels, allowed_pairs, n_examples, label_index):
    """
    Sample *n_examples* (img1, img2) pairs whose digit labels are drawn
    exclusively from *allowed_pairs*.
    """
    imgs_1, imgs_2 = [], []
    lab_1,  lab_2  = [], []

    for _ in range(n_examples):
        d1, d2 = random.choice(allowed_pairs)
        i1 = random.choice(label_index[d1])
        i2 = random.choice(label_index[d2])
        imgs_1.append(imgs[i1])
        imgs_2.append(imgs[i2])
        lab_1.append(d1)
        lab_2.append(d2)

    return (
        torch.stack(imgs_1),
        torch.stack(imgs_2),
        torch.tensor(lab_1),
        torch.tensor(lab_2),
    )


def _build_test_pairs(labels: torch.Tensor) -> list[tuple[int, int]]:
    """
    All mixed-parity (even, odd) and (odd, even) pairs from digits present
    in *labels*, excluding any pair that appears in TRAIN_PAIRS.
    """
    train_set = set(TRAIN_PAIRS)
    digits = sorted(set(labels.tolist()))
    evens = [d for d in digits if d % 2 == 0]
    odds  = [d for d in digits if d % 2 == 1]

    pairs = []
    for e in evens:
        for o in odds:
            for p in [(e, o), (o, e)]:
                if p not in train_set:
                    pairs.append(p)
    return pairs


def get_mnist_evenodd_dataloader(n_train, n_test, b_size):
    """
    Build the MNIST-EvenOdd dataset as described in the paper.

    Training  : 6 720 same-parity pairs drawn from the 16 restricted pairs in
                Table 6 (8 even+even, 8 odd+odd).
    Validation: not returned here (see note below).
    Test      : 960 out-of-distribution mixed-parity (even+odd / odd+even)
                pairs that are absent from the training support.

    Returns
    -------
    train_loader, test_loader, anchor_imgs
    """
    # ── Load raw MNIST ──────────────────────────────────────────────────────
    mnist_train = torchvision.datasets.MNIST(
        "src/datasets/", train=True,  download=True,
        transform=torchvision.transforms.ToTensor()
    )
    mnist_test  = torchvision.datasets.MNIST(
        "src/datasets/",  train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )

    train_imgs,  train_labels = mnist_train.data.float() / 255.0, mnist_train.targets
    test_imgs,   test_labels  = mnist_test.data.float()  / 255.0, mnist_test.targets

    # Add channel dimension: (N, 1, H, W)
    train_imgs = train_imgs.unsqueeze(1)
    test_imgs  = test_imgs.unsqueeze(1)

    # ── Anchor images (one per digit class) ────────────────────────────────
    anchor_imgs = get_anchors(train_imgs, train_labels)

    # ── Build label → index maps ────────────────────────────────────────────
    train_index = _build_label_index(train_labels)
    test_index  = _build_label_index(test_labels)

    # ── Training pairs: restricted to TRAIN_PAIRS (same parity only) ───────
    train_img1, train_img2, train_lab1, train_lab2 = _sample_from_pairs(
        train_imgs, train_labels, TRAIN_PAIRS, n_train, train_index
    )

    # ── Test pairs: out-of-distribution mixed-parity pairs ──────────────────
    test_pairs = _build_test_pairs(test_labels)
    test_img1,  test_img2,  test_lab1,  test_lab2  = _sample_from_pairs(
        test_imgs, test_labels, test_pairs, n_test, test_index
    )

    # ── Pack into TensorDatasets ─────────────────────────────────────────────
    def _make_dataset(img1, img2, lab1, lab2, train=False):
        imgs_operand    = torch.stack([img1, img2], dim=1)   # (N, 2, 1, H, W)
        label_addition  = lab1 + lab2
        idx             = torch.arange(len(lab1))
        if train:
            return torch.utils.data.TensorDataset(
                imgs_operand, label_addition, idx
            )
        else:
            return torch.utils.data.TensorDataset(
                imgs_operand, label_addition, lab1, lab2, idx
            )

    train_set = _make_dataset(train_img1, train_img2, train_lab1, train_lab2, train=True)
    test_set  = _make_dataset(test_img1,  test_img2,  test_lab1,  test_lab2)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=b_size, shuffle=True
    )
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=b_size, shuffle=False
    )

    return train_loader, test_loader, anchor_imgs


# ── Unchanged addition dataloader (kept for compatibility) ──────────────────
def get_mnist_addition_dataloader(n_train, n_test, b_size):
    n_operands = 2

    mnist_train = torchvision.datasets.MNIST(
        "src/datasets/", train=True,  download=True,
        transform=torchvision.transforms.ToTensor()
    )
    mnist_test  = torchvision.datasets.MNIST(
        "src/datasets/",  train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )

    train_imgs, train_labels = mnist_train.data, mnist_train.targets
    test_imgs,  test_labels  = mnist_test.data,  mnist_test.targets

    train_imgs = torch.unsqueeze(train_imgs / 255.0, 1)
    test_imgs  = torch.unsqueeze(test_imgs  / 255.0, 1)

    anchor_imgs = get_anchors(train_imgs, train_labels)

    imgs_operand_train    = [train_imgs  [i * n_train:i * n_train + n_train] for i in range(n_operands)]
    labels_operand_train  = [train_labels[i * n_train:i * n_train + n_train] for i in range(n_operands)]
    imgs_operand_test     = [test_imgs   [i * n_test :i * n_test  + n_test]  for i in range(n_operands)]
    labels_operand_test   = [test_labels [i * n_test :i * n_test  + n_test]  for i in range(n_operands)]

    label_addition_train  = labels_operand_train[0] + labels_operand_train[1]
    label_addition_test   = labels_operand_test[0]  + labels_operand_test[1]
    label_mult_train      = labels_operand_train[0] * labels_operand_train[1]
    label_mult_test       = labels_operand_test[0]  * labels_operand_test[1]

    train_set = torch.utils.data.TensorDataset(
        torch.stack(imgs_operand_train, dim=1), label_addition_train,
        torch.arange(n_train)
    )
    test_set  = torch.utils.data.TensorDataset(
        torch.stack(imgs_operand_test,  dim=1), label_addition_test,
        labels_operand_test[0],  labels_operand_test[1],
        torch.arange(n_test)
    )

    return (
        torch.utils.data.DataLoader(train_set, batch_size=b_size, shuffle=True),
        torch.utils.data.DataLoader(test_set,  batch_size=b_size, shuffle=False),
        anchor_imgs,
    )


if __name__ == '__main__':
    train_loader, test_loader, anchor = get_mnist_evenodd_dataloader(
        n_train=6720, n_test=960, b_size=64
    )

    images, label_add, idx = next(iter(train_loader))
    _, _, label_d1, label_d2, _ = next(iter(test_loader))

    print("Batch shapes  :", images.shape, label_add.shape)

    # Spot-check test set: all mixed-parity
    _, _, td1, td2, _ = next(iter(test_loader))
    assert all(
        (d1 % 2) != (d2 % 2) for d1, d2 in zip(td1.tolist(), td2.tolist())
    ), "Test set contains same-parity pairs!"
    print("All test pairs mixed-parity (OOD) ✓")