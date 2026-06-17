import random
from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, TensorDataset


from src.backbones.Sudoku4x4.protoencoder import (
    ProtoEncoder, compute_prototypes, prototypical_loss, split_support_query, augment_support_set, digit_probs
    )


from src.models.Sudoku4x4.helpers.sudoku4x4_sl import semantic_loss_sudoku, _prob_group_is_permutation, _sudoku_groups

NUM_DIGITS  = 4          # digit classes: 1, 2, 3, 4
BOARD_CELLS = 16         # 4 × 4

_GROUPS = _sudoku_groups()

# ── 7. TRAINING LOOP (Algorithm 1) ───────────────────────────────────────────

def train_one_epoch(
    encoder:      ProtoEncoder,
    optimizer:    torch.optim.Optimizer,
    aug_images:   torch.Tensor,       # augmented support (40, 1, 28, 28)
    aug_labels:   torch.Tensor,       # (40,)  values in {1,2,3,4}
    unsup_loader: DataLoader,         # yields (imgs, y, boards, idx)
    proto_images: torch.Tensor,       # clean 1-per-class support (4, 1, 28, 28)
    proto_labels: torch.Tensor,       # (4,)  values in {1,2,3,4}
    device:       torch.device,
    episodes:     int   = 50,
    k_support:    int   = 1,
    sl_weight:    float = 10.0,
) -> dict:
    encoder.train()
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)

    # ── Phase 1: episodic warm-up — accumulate proto loss, NO optimiser step ─
    # Algorithm 1 computes a single combined loss (proto + NeSy) per board
    # batch and takes one gradient step. Running 50 independent optimiser
    # steps here (as in the original) would bias the encoder toward prototypical
    # loss before any NeSy signal is seen. Instead we accumulate the episodes
    # into a warm-up loss that is added once to the first Phase-2 batch.
    ep_losses, ep_accs = [], []
    warmup_loss = torch.tensor(0.0, device=device)
    for _ in range(episodes):
        s_img, s_lbl, q_img, q_lbl = split_support_query(aug_images, aug_labels, k_support)
        s_img, s_lbl = s_img.to(device), s_lbl.to(device)
        q_img, q_lbl = q_img.to(device), q_lbl.to(device)

        all_imgs = torch.cat([s_img, q_img])
        all_lbls = torch.cat([s_lbl, q_lbl])
        embs     = encoder(all_imgs)

        loss, acc = prototypical_loss(embs, all_lbls, n_support=k_support)
        warmup_loss = warmup_loss + loss
        ep_losses.append(loss.item())
        ep_accs.append(acc.item())

    # ── Phase 2: joint proto + Semantic Loss on unsupervised boards ──────────
    # DataLoader tuple order (from dataloader_sudoku4x4r.py TensorDataset):
    #   (imgs, labels, boards, idx)
    #   imgs   : (B, 4, 4, 1, 28, 28)  — cell images (4×4 grid layout)
    #   labels : (B,)                   — 0 or 1
    #   boards : (B, 4, 4)              — integer digit values (unused in training)
    #   idx    : (B,)                   — sample indices (unused)
    nesy_losses, nesy_sl, nesy_proto = [], [], []
    first_batch = True

    for (imgs, y_label, _boards, _idx) in unsup_loader:
        # FIX 1: correct unpack order — imgs first, then y_label.
        # FIX 2: reshape (B,4,4,1,28,28) → (B,16,1,28,28) explicitly;
        #        .view() on a non-contiguous tensor would crash, .reshape()
        #        is safe regardless of memory layout.
        imgs    = imgs.to(device)                                  # (B, 4, 4, 1, 28, 28)
        y_label = y_label.to(device).long().view(-1)               # (B,)
        B       = imgs.size(0)
        boards  = imgs.reshape(B, BOARD_CELLS, 1, 28, 28)         # (B, 16, 1, 28, 28)

        optimizer.zero_grad()

        # Re-compute prototypes so gradients flow (Theorem 4.1)
        prototypes = compute_prototypes(proto_images, proto_labels, encoder)  # (4, D)

        # Embed all 16 cells of all B boards in one forward pass
        boards_flat = boards.reshape(B * BOARD_CELLS, 1, 28, 28)  # (B*16, 1, 28, 28)
        embs_flat   = encoder(boards_flat)                         # (B*16, D)

        # Compute P(digit 1..4) for each cell
        probs_flat  = digit_probs(embs_flat, prototypes)           # (B*16, 4)
        cell_probs  = probs_flat.view(B, BOARD_CELLS, NUM_DIGITS)  # (B, 16, 4)

        # Semantic Loss
        loss_sl = semantic_loss_sudoku(cell_probs, y_label)

        # Mini-episode prototypical loss (Algorithm 1, lines 8-13):
        # combined in the SAME gradient step as the NeSy loss.
        s_img, s_lbl, q_img, q_lbl = split_support_query(aug_images, aug_labels, k_support)
        s_img, s_lbl = s_img.to(device), s_lbl.to(device)
        q_img, q_lbl = q_img.to(device), q_lbl.to(device)
        ep_emb = encoder(torch.cat([s_img, q_img]))
        ep_lbl = torch.cat([s_lbl, q_lbl])
        loss_proto, _ = prototypical_loss(ep_emb, ep_lbl, n_support=k_support)

        # FIX 3: on the first batch, fold in the Phase-1 warm-up loss so that
        # the warm-up episodes contribute exactly one gradient step rather than
        # 50 independent steps (as in the original code).
        loss = sl_weight * loss_sl + loss_proto
        if first_batch:
            loss = loss + warmup_loss
            first_batch = False

        loss.backward()
        optimizer.step()

        nesy_losses.append(loss.item())
        nesy_sl.append(loss_sl.item())
        nesy_proto.append(loss_proto.item())

    return {
        "ep_loss":    sum(ep_losses)    / max(len(ep_losses),    1),
        "ep_acc":     sum(ep_accs)      / max(len(ep_accs),      1),
        "nesy_loss":  sum(nesy_losses)  / max(len(nesy_losses),  1),
        "nesy_sl":    sum(nesy_sl)      / max(len(nesy_sl),      1),
        "nesy_proto": sum(nesy_proto)   / max(len(nesy_proto),   1),
    }


# ── 8. EVALUATION ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    encoder:      ProtoEncoder,
    test_loader:  DataLoader,    # yields (imgs, y_label, boards, idx)
    proto_images: torch.Tensor,
    proto_labels: torch.Tensor,
    device:       torch.device,
    sampler
) -> dict:
    """
    Metrics (mirrors Table 1 of the paper):
      digit_acc  — Acc(C): fraction of individual cell digits correct
      digit_f1   — F1(C):  macro-averaged F1 over the 4 digit classes
      board_acc  — Acc(Y): fraction of boards whose validity label is correct
      cls_c      — Cls(C): fraction of digit classes that collapse to one
                            predicted class (0 = no collapse, 1 = full collapse)
    """
    encoder.eval()
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)
    prototypes   = compute_prototypes(proto_images, proto_labels, encoder)

    tp = torch.zeros(NUM_DIGITS)
    fp = torch.zeros(NUM_DIGITS)
    fn = torch.zeros(NUM_DIGITS)

    board_tp = torch.tensor(0.0, device=device)
    board_fp = torch.tensor(0.0, device=device)
    board_fn = torch.tensor(0.0, device=device)
    
    correct_digit, total_digits       = 0, 0
    correct_board,  total_boards      = 0, 0
    all_preds: list[torch.Tensor]     = []

    # DataLoader tuple order: (imgs, y_label, boards, idx)
    #   imgs    : (B, 4, 4, 1, 28, 28)
    #   y_label : (B,)        0 or 1
    #   boards  : (B, 4, 4)   integer digit values — used as true_digits
    #   idx     : (B,)        sample indices (unused)
    for (imgs, y_label, true_digits, _idx) in test_loader:
        # FIX 1 + 2: correct unpack order and explicit reshape.
        imgs        = imgs.to(device)                              # (B, 4, 4, 1, 28, 28)
        y_label     = y_label.to(device).long().view(-1)           # (B,)
        # boards tensor is (B, 4, 4); flatten to (B, 16) for per-cell comparison.
        true_digits = true_digits.to(device).view(-1, BOARD_CELLS) # (B, 16)

        B           = imgs.size(0)
        boards      = imgs.reshape(B, BOARD_CELLS, 1, 28, 28)     # (B, 16, 1, 28, 28)

        boards_flat = boards.reshape(B * BOARD_CELLS, 1, 28, 28)
        embs_flat   = encoder(boards_flat)
        probs_flat  = digit_probs(embs_flat, prototypes)           # (B*16, 4)

        # argmax gives index 0-3; add 1 to get digit 1-4
        pred_flat   = probs_flat.argmax(1) + 1                     # (B*16,)  in {1,2,3,4}
        pred_digits = pred_flat.view(B, BOARD_CELLS)               # (B, 16)
        sq_pred_digits = pred_flat.view(B, 4, 4) 
        all_preds.append(pred_flat.cpu())

        # ── digit accuracy ───────────────────────────────────────────────────
        correct_digit += (pred_digits.view(-1) == true_digits.view(-1)).sum().item()
        total_digits  += B * BOARD_CELLS

        # ── per-class TP / FP / FN for macro F1 ─────────────────────────────
        true_flat = true_digits.view(-1)       # (B*16,)
        for d in range(1, NUM_DIGITS + 1):
            i = d - 1
            tp[i] += ((pred_flat == d) & (true_flat == d)).sum().item()
            fp[i] += ((pred_flat == d) & (true_flat != d)).sum().item()
            fn[i] += ((pred_flat != d) & (true_flat == d)).sum().item()

        # ── board validity accuracy ───────────────────────────────────────────
        cell_probs  = probs_flat.view(B, BOARD_CELLS, NUM_DIGITS)

        log_p_valid = torch.zeros(B, device=device)
        for group in _GROUPS:
            gp     = cell_probs[:, group, :]
            p_perm = _prob_group_is_permutation(gp)
            log_p_valid += torch.log(p_perm.clamp(min=1e-12))
        p_valid  = torch.exp(log_p_valid.clamp(max=0.0))
        pred_y   = (p_valid >= 0.5).long()

        sq_pred_y = sampler.tensor_check(sq_pred_digits).to(device)

        correct_board += (sq_pred_y == y_label).sum().item()
        total_boards  += B

        board_tp += ((sq_pred_y == 1) & (y_label == 1)).sum()
        board_fp += ((sq_pred_y == 1) & (y_label == 0)).sum()
        board_fn += ((sq_pred_y == 0) & (y_label == 1)).sum()

    # ── macro F1 ─────────────────────────────────────────────────────────────
    precision = tp / (tp + fp).clamp(min=1e-8)
    recall    = tp / (tp + fn).clamp(min=1e-8)
    f1_per    = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    macro_f1  = f1_per.mean().item()

    board_precision = board_tp / (board_tp + board_fp).clamp(min=1e-8) #type:ignore
    board_recall    = board_tp / (board_tp + board_fn).clamp(min=1e-8) #type:ignore

    board_f1 = (
        2 * board_precision * board_recall
        / (board_precision + board_recall).clamp(min=1e-8)
    ).item()

    # ── concept collapse ─────────────────────────────────────────────────────
    all_preds_t      = torch.cat(all_preds)                      # (N*16,)
    predicted_classes = torch.unique(all_preds_t).numel()
    cls_c             = 1.0 - predicted_classes / NUM_DIGITS

    return {
        "digit_acc": 100.0 * correct_digit / max(total_digits, 1),
        "digit_f1":  100.0 * macro_f1,
        "board_acc": 100.0 * correct_board / max(total_boards, 1),
        "board_f1":  100.0 * board_f1,
        "cls_c":     cls_c,
    }


# ── 9. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    seed = 0
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    random.seed(seed)

    from src.samplers.sudoku4x4_sampler import Sampler
    sampler = Sampler()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    from src.databuilder.Sudoku4x4.databuilder import get_mnist_sudoku4x4_dataloader
    train_loader, test_loader, osat_train_loader, _, proto_images = \
        get_mnist_sudoku4x4_dataloader(device, n_train=300, n_test=1000, batch_size=64)

    proto_labels = torch.arange(len(proto_images)) + 1   # tensor([1, 2, 3, 4])

    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)
    print(f"Proto labels: {proto_labels.tolist()}")

    # ── Augment 1-per-class support set → 40 images ──────────────────────────
    aug_images, aug_labels = augment_support_set(
        proto_images, proto_labels, device, num_augmentations=9
    )
    print(f"Augmented support set: {aug_images.shape[0]} images, "
          f"{torch.unique(aug_labels).numel()} classes "
          f"(labels: {torch.unique(aug_labels).tolist()})")

    # ── Model & optimiser ─────────────────────────────────────────────────────
    encoder   = ProtoEncoder(x_dim=1, hid_dim=64, z_dim=64).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # ── Training hyper-parameters (Appendix E of the paper) ──────────────────
    num_epochs = 20
    sl_weight  = 10.0    # wsl
    episodes   = 50

    best_board_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        stats = train_one_epoch(
            encoder      = encoder,
            optimizer    = optimizer,
            aug_images   = aug_images,
            aug_labels   = aug_labels,
            unsup_loader = osat_train_loader,
            proto_images = proto_images,
            proto_labels = proto_labels,
            device       = device,
            episodes     = episodes,
            k_support    = 1,
            sl_weight    = sl_weight,
        )
        scheduler.step()

        ev = evaluate(encoder, test_loader, proto_images, proto_labels, device, sampler)

        print(
            f"(PNet + SL) Epoch {epoch:02d}/{num_epochs} | "
            f"ep_loss={stats['ep_loss']:.4f} ep_acc={stats['ep_acc']:.4f} | "
            f"nesy_loss={stats['nesy_loss']:.4f} "
            f"(sl={stats['nesy_sl']:.4f} proto={stats['nesy_proto']:.4f}) | "
            f"digit_acc={ev['digit_acc']:.1f}% digit_f1={ev['digit_f1']:.1f}% "
            f"board_acc={ev['board_acc']:.1f}% board_f1={ev['board_f1']:.1f}% cls_c={ev['cls_c']:.3f}"
        )

        if ev["board_acc"] > best_board_acc:
            best_board_acc = ev["board_acc"]
            os.makedirs("src/models/Sudoku4x4/checkpoints", exist_ok=True)
            torch.save(encoder.state_dict(), "src/models/Sudoku4x4/checkpoints/best_encoder_sudoku4x4.pth")
            print(f"  → Saved best model (digit_acc={ev['digit_acc']:.1f}%,  "
                  f"board_acc={best_board_acc:.1f}%, "
                  f"F1(C)={ev['digit_f1']:.1f}%, "
                  f"F1(Y)={ev['board_f1']:.1f}%, "
                  f"Cls(C)={ev['cls_c']:.3f})")

    print(f"\nTraining complete. Best board accuracy: {best_board_acc:.1f}%")