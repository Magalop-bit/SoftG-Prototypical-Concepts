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

from src.models.Sudoku4x4.helpers.sudoku4x4_sl import _prob_group_is_permutation, _sudoku_groups
from src.samplers.sudoku4x4_sampler import Sampler


NUM_DIGITS  = 4          # digit classes: 1, 2, 3, 4
BOARD_CELLS = 16         # 4 × 4

_GROUPS = _sudoku_groups()

# ── 7. TRAINING LOOP (Algorithm 1) ───────────────────────────────────────────

def train_one_epoch(
    encoder:         ProtoEncoder,
    optimizer:       torch.optim.Optimizer,
    # --- episodic (augmented support set) ---
    aug_images:      torch.Tensor,
    aug_labels:      torch.Tensor,
    # --- unsupervised NeSy dataloader ---
    unsup_loader:    DataLoader,
    # --- clean (1-per-class) support set for NeSy prototypes ---
    proto_images:    torch.Tensor,
    proto_labels:    torch.Tensor,
    device:          torch.device,
    # softg-parameters
    candidate_cache: torch.Tensor,
    sampler:         Sampler,
    K:               int   = 1,
    sampling_epoch:  int   = 1,
    T:               float = 1.0,
    # hyper-parameters
    episodes:        int   = 50,
    k_support:       int   = 1,
    sl_weight:       float = 10.0,
    epoch:           int   = 0,
) -> dict:
    encoder.train()
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # ── Phase 1: episodic warm-up — accumulate proto loss, NO optimiser step ─
    # Algorithm 1 computes a single combined loss (proto + NeSy) per board
    # batch and takes one gradient step. Running 50 independent optimiser
    # steps here (as in the original) would bias the encoder toward prototypical
    # loss before any NeSy signal is seen. Instead we accumulate the episodes
    # into a warm-up loss that is added once to the first Phase-2 batch.
    ep_losses, ep_accs = [], []
    warmup_loss = torch.tensor(0.0, device=device)
    optimizer.zero_grad()
    for _ in range(episodes):
        optimizer.zero_grad()
        s_img, s_lbl, q_img, q_lbl = split_support_query(aug_images, aug_labels, k_support)
        s_img, s_lbl = s_img.to(device), s_lbl.to(device)
        q_img, q_lbl = q_img.to(device), q_lbl.to(device)
        embs       = encoder(torch.cat([s_img, q_img]))
        loss, acc  = prototypical_loss(embs, torch.cat([s_lbl, q_lbl]), n_support=k_support)
        warmup_loss = warmup_loss + loss
        ep_losses.append(loss.item())
        ep_accs.append(acc.item())
        loss.backward()
        optimizer.step()

    # ── Phase 2: joint proto + Semantic Loss on unsupervised boards ──────────
    # DataLoader tuple order (from dataloader_sudoku4x4r.py TensorDataset):
    #   (imgs, labels, boards, idx)
    #   imgs   : (B, 4, 4, 1, 28, 28)  — cell images (4×4 grid layout)
    #   labels : (B,)                   — 0 or 1
    #   boards : (B, 4, 4)              — integer digit values (unused in training)
    #   idx    : (B,)                   — sample indices (unused)
    nesy_losses, nesy_kl, nesy_proto = [], [], []
    first_batch = True

    for (imgs, y_label, _boards, idx) in unsup_loader:
        # FIX 1: correct unpack order — imgs first, then y_label.
        # FIX 2: reshape (B,4,4,1,28,28) → (B,16,1,28,28) explicitly;
        #        .view() on a non-contiguous tensor would crash, .reshape()
        #        is safe regardless of memory layout.
        imgs    = imgs.to(device)                                  # (B, 4, 4, 1, 28, 28)
        y_label = y_label.to(device).long().view(-1)               # (B,)
        idx     = idx.to(device)
        B       = imgs.size(0)
        boards  = imgs.reshape(B, BOARD_CELLS, 1, 28, 28)         # (B, 16, 1, 28, 28)

        # Re-compute prototypes so gradients flow (Theorem 4.1)
        prototypes = compute_prototypes(proto_images, proto_labels, encoder)  # (4, D)

        # Embed all 16 cells of all B boards in one forward pass
        boards_flat = boards.reshape(B * BOARD_CELLS, 1, 28, 28)  # (B*16, 1, 28, 28)
        embs_flat   = encoder(boards_flat)                         # (B*16, D)

        # Compute P(digit 1..4) for each cell
        probs_flat  = digit_probs(embs_flat, prototypes)           # (B*16, 4)
        cell_probs  = probs_flat.view(B, BOARD_CELLS, NUM_DIGITS)  # (B, 16, 4)

        flat_logits    = -(torch.cdist(embs_flat, prototypes, p=2) ** 2)                    # [B*16, 4]
        flat_sq_logits = flat_logits.view(B, 4 * 4, -1)       # [B, 16, 4]

        # ── Update population cache ───────────────────────────────────────
        if epoch == 0:
            with torch.no_grad():
                batch_k_sample = sampler.batch_k_sample(device, y_label, batch_size=B, k=K*10)  # [B, K, 4, 4]
                scores   = compute_score_baseline(flat_sq_logits.detach(), batch_k_sample)
                best_values, best = torch.topk(scores, K, dim=1)
                candidate_cache[idx] = batch_k_sample[torch.arange(B, device=device).unsqueeze(1), best]  # [B, K, 2]
        elif (epoch % sampling_epoch == 0) and (epoch != 0):
            # Refresh: draw K fresh valid/invalid candidates
            with torch.no_grad():
                population = candidate_cache[idx]
                candidate_cache[idx] = metropolis_walk_population(
                    sampler, flat_sq_logits.detach(), population, y_label, T
                )
        else:
            pass
        
        population = candidate_cache[idx]                          # [B, K, 4, 4]

        with torch.no_grad():
            raw_scores = compute_score_baseline(flat_sq_logits.detach(), population)  # [B, K]
            weights = torch.softmax(raw_scores / T, dim=1)         # [B, K]

        # Compute per-cell soft target by marginalizing over candidates
        # soft_target[b, cell, digit] = Σ_k w_k * 1[population[b,k,cell] == digit]
        flat_pop = population.view(B, K, BOARD_CELLS)              # [B, K, 16]
        soft_targets = torch.zeros(B, BOARD_CELLS, NUM_DIGITS, device=device)
        for d in range(1, NUM_DIGITS + 1):
            soft_targets[:, :, d-1] = (
                weights.unsqueeze(-1) * (flat_pop == d).float()
            ).sum(dim=1)                                           # [B, 16]

        # KL divergence between model distribution and soft target
        log_probs  = F.log_softmax(flat_sq_logits, dim=-1)         # [B, 16, 4]
        loss_kl    = F.kl_div(log_probs, soft_targets, reduction='batchmean')
        loss_kl    = loss_kl / BOARD_CELLS

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
        loss = sl_weight * loss_kl + loss_proto
        if False:
            loss = loss + warmup_loss
            first_batch = False

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        nesy_losses.append(loss.item())
        nesy_kl.append(loss_kl.item())
        nesy_proto.append(loss_proto.item())

    return {
        "ep_loss":    sum(ep_losses)    / max(len(ep_losses),    1),
        "ep_acc":     sum(ep_accs)      / max(len(ep_accs),      1),
        "nesy_loss":  sum(nesy_losses)  / max(len(nesy_losses),  1),
        "nesy_kl":    sum(nesy_kl)      / max(len(nesy_kl),      1),
        "nesy_proto": sum(nesy_proto)   / max(len(nesy_proto),   1),
    }

# ── Stage-II: fine-tune with deterministic argmax grounding (backward stage) ──
def train_softg_backward(model, sampler, device, train_loader, proto_images, proto_labels, optimizer, K=5):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = []

    for batch_idx, (images, board_labels, _, idxs) in enumerate(train_loader):
        images       = images.to(device)
        board_labels = board_labels.to(device)
        B, N, _, C, H, W = images.shape

        flat_images       = images.view(B * N * N, C, H, W)

        flat_embd    = model(flat_images)                    # [B*16, 4]

        prototypes = compute_prototypes(proto_images, proto_labels, encoder)  # (4, D)

        flat_logits    = -(torch.cdist(flat_embd, prototypes, p=2) ** 2)                    # [B*16, 4]
        flat_sq_logits = flat_logits.view(B, 4 * 4, -1)       # [B, 16, 4]

        with torch.no_grad():
            population = sampler.batch_k_sample(
                board_labels, batch_size=B, k=K*5
            )

        scores   = compute_score_baseline(flat_sq_logits, population)  # [B, K]
        best = scores.argmax(dim=1)
        best_boards = population[torch.arange(B, device=device), best]

        # Mini-episode prototypical loss (Algorithm 1, lines 8-13):
        # combined in the SAME gradient step as the NeSy loss.
        s_img, s_lbl, q_img, q_lbl = split_support_query(aug_images, aug_labels, k_support=1)
        s_img, s_lbl = s_img.to(device), s_lbl.to(device)
        q_img, q_lbl = q_img.to(device), q_lbl.to(device)
        ep_emb = encoder(torch.cat([s_img, q_img]))
        ep_lbl = torch.cat([s_lbl, q_lbl])
        loss_proto, _ = prototypical_loss(ep_emb, ep_lbl, n_support=1)

        loss = 5.0 * criterion(flat_logits, best_boards.view(-1) - 1) + loss_proto

        #loss = joint_criteria(logits, best_digits)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    return {
        "nesy_loss":  sum(total_loss) / max(len(total_loss),1),
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

def compute_score_baseline(logits, boards):
    """
    Joint log-probability score for each candidate board.

    Args:
        logits: [B, 16, 4]   — per-cell class logits
        boards: [B, K, 4, 4] — candidate boards (values 1..4)

    Returns:
        scores: [B, K]        — sum of log-probs over all 16 cells
    """
    B, K, _, _ = boards.shape
    # expand logits to [B, K, 16, 4]
    logits_exp = logits.unsqueeze(1).expand(-1, K, -1, -1)
    log_probs  = torch.log_softmax(logits_exp, dim=-1)

    flat_boards = boards.clone().view(B, K, -1) - 1          # [B, K, 16]  (0-indexed)
    chosen = log_probs.gather(
        dim=3,
        index=flat_boards.unsqueeze(-1)
    ).squeeze(-1)                                             # [B, K, 16]

    return chosen.sum(dim=-1)                                 # [B, K]


def metropolis_walk_population(sampler, logits_board, population, board_labels, T):
    """
    Apply one Metropolis step independently to every candidate in the population.

    Args:
        logits_board: [B, 16, 4]   — current model logits (detached)
        population:   [B, K, 4, 4] — current population of candidate boards
        board_labels: [B]           — 1=SAT, 0=UNSAT per example
        T:            float         — temperature (gamma in the paper)

    Returns:
        new_population: [B, K, 4, 4] — updated population after accept/reject
    """
    with torch.no_grad():
        B, K, _, _ = population.shape
        device = population.device

        # Flatten [B, K, 4, 4] → [B*K, 4, 4] so sampler.walk sees a flat batch
        flat_pop    = population.view(B * K, 4, 4)
        # Repeat board_labels K times: [B] → [B*K]
        flat_labels = board_labels.unsqueeze(1).expand(B, K).reshape(B * K)

        # One walk step per candidate  →  [B*K, 4, 4]
        flat_proposed = sampler.walk(flat_pop, flat_labels)

        flat_proposed = flat_proposed.view(B * K, 4, 4)
        # Reshape back to [B, K, 4, 4]
        proposed = flat_proposed.view(B, K, 4, 4)

        # Score old and proposed populations
        old_scores = compute_score_baseline(logits_board, population)   # [B, K]
        new_scores = compute_score_baseline(logits_board, proposed)     # [B, K]

        # Metropolis acceptance: log u < (new - old) / T
        tau    = (new_scores - old_scores) / T                          # [B, K]

        if T != 0.0:
            accept = torch.log(torch.rand_like(tau)) < tau                  # [B, K] bool
        else:
            accept = new_scores > old_scores

        # Where accepted, take proposal; otherwise keep old
        accept_expanded = accept.unsqueeze(-1).unsqueeze(-1)            # [B, K, 1, 1]
        new_population  = torch.where(
            accept_expanded.expand_as(proposed), proposed, population
        )

    return new_population                                               # [B, K, 4, 4]


# ── 9. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    seed = 0
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    random.seed(seed)

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
    checkpoint_path = "src/SoftPNet/Sudoku4x4/checkpoints/best_encoder_sudoku4x4.pth"
    checkpoint_path_bs = "src/SoftPNet/Sudoku4x4/checkpoints/best_encoder_sudoku4x4_bs.pth"

    T0 = T = 1.0
    t = 1
    K = 100
    schedule = 'exp'
    sampling_epoch = 2

    candidate_cache = torch.zeros((len(train_loader.dataset), K, 4, 4), dtype=torch.long).to(device) #type:ignore

    # ── Training hyper-parameters (Appendix E of the paper) ──────────────────
    num_epochs = 20
    sl_weight  = 15.0    # wsl
    episodes   = 50

    best_board_acc = 0.0
    for epoch in range(num_epochs):
        stats = train_one_epoch(
            encoder      = encoder,
            optimizer    = optimizer,
            aug_images   = aug_images,
            aug_labels   = aug_labels,
            unsup_loader = osat_train_loader,
            proto_images = proto_images,
            proto_labels = proto_labels,
            device       = device,
            candidate_cache=candidate_cache,
            K            = K,
            sampling_epoch= sampling_epoch,
            T            = T,
            sampler      = sampler,  
            episodes     = episodes,
            k_support    = 1,
            sl_weight    = sl_weight,
            epoch        = epoch,
        )
        scheduler.step()

        ev = evaluate(encoder, test_loader, proto_images, proto_labels, device, sampler)

        print(
            f"(Soft-PNet) Epoch {epoch+1:02d}/{num_epochs} | "
            f"ep_loss={stats['ep_loss']:.4f} ep_acc={stats['ep_acc']:.4f} | "
            f"nesy_loss={stats['nesy_loss']:.4f} "
            f"(kl={stats['nesy_kl']:.4f} proto={stats['nesy_proto']:.4f}) | "
            f"digit_acc={ev['digit_acc']:.1f}% digit_f1={ev['digit_f1']:.1f}% "
            f"board_acc={ev['board_acc']:.1f}% board_f1={ev['board_f1']:.1f}% cls_c={ev['cls_c']:.3f}"
        )

        if ev["board_acc"] > best_board_acc:
            best_board_acc = ev["board_acc"]
            os.makedirs("src/SoftPNet/Sudoku4x4/checkpoints", exist_ok=True)
            torch.save(encoder.state_dict(), checkpoint_path)
            print(f"  → Saved best model (digit_acc={ev['digit_acc']:.1f}%,  "
                  f"board_acc={best_board_acc:.1f}%, "
                  f"F1(C)={ev['digit_f1']:.1f}%, "
                  f"F1(Y)={ev['board_f1']:.1f}%, "
                  f"Cls(C)={ev['cls_c']:.3f})")

    print(f"\nTraining complete. Best board accuracy: {best_board_acc:.1f}%")