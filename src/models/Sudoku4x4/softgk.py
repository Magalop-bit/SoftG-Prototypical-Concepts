import math
import random
import numpy as np
import torch

from src.backbones.Sudoku4x4.lenet import LeNet
from src.databuilder.Sudoku4x4.databuilder import get_mnist_sudoku4x4_dataloader
from src.samplers.sudoku4x4_sampler import Sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sampler = Sampler()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Scoring ────────────────────────────────────────────────────────────────────

def compute_score_baseline(logits, boards):
    """
    Joint log-probability score for each candidate board.

    Args:
        logits: [B, 16, 4]   — per-cell class logits
        boards: [B, K, 4, 4] — candidate boards (values 1..4)

    Returns:
        scores: [B, K]        — sum of log-probs over all 16 cells
    """
    with torch.no_grad():
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


# ── Population-level Metropolis walk ──────────────────────────────────────────

def metropolis_walk_population(logits_board, population, board_labels, T):
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

        # Flatten [B, K, 4, 4] → [B*K, 4, 4] so sampler.walk sees a flat batch
        flat_pop    = population.view(B * K, 4, 4)
        # Repeat board_labels K times: [B] → [B*K]
        flat_labels = board_labels.unsqueeze(1).expand(B, K).reshape(B * K)

        # One walk step per candidate  →  [B*K, 4, 4]
        flat_proposed = sampler.walk(flat_pop, flat_labels)

        # Reshape back to [B, K, 4, 4]
        proposed = flat_proposed.view(B, K, 4, 4)

        # Score old and proposed populations
        old_scores = compute_score_baseline(logits_board, population)   # [B, K]
        new_scores = compute_score_baseline(logits_board, proposed)     # [B, K]

        # Metropolis acceptance: log u < (new - old) / T
        tau    = (new_scores - old_scores) / T                          # [B, K]
        accept = torch.log(torch.rand_like(tau)) < tau                  # [B, K] bool

        # Where accepted, take proposal; otherwise keep old
        accept_expanded = accept.unsqueeze(-1).unsqueeze(-1)            # [B, K, 1, 1]
        new_population  = torch.where(
            accept_expanded.expand_as(proposed), proposed, population
        )

    return new_population                                               # [B, K, 4, 4]


# ── Top-k loss ─────────────────────────────────────────────────────────────────

def topk_ce_loss(flat_logits, population, flat_sq_logits, k_loss):
    """
    Score all K candidates, select the top-k_loss per example, compute
    mean CE loss over those B * k_loss pseudo-labels.

    Args:
        flat_logits:    [B*16, 4]  — raw logits from the network
        population:     [B, K, 4, 4]
        flat_sq_logits: [B, 16, 4] — same logits reshaped (for scoring)
        k_loss:         int        — how many top candidates to use per example

    Returns:
        loss: scalar
    """
    B, K, _, _ = population.shape

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        scores  = compute_score_baseline(flat_sq_logits, population)    # [B, K]
        # top-k indices per example
        topk_idx = scores.topk(k_loss, dim=1).indices                   # [B, k_loss]

        # Gather the top-k boards  →  [B, k_loss, 4, 4]
        topk_idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(
            B, k_loss, 4, 4
        )
        best_boards = population.gather(dim=1, index=topk_idx_exp)      # [B, k_loss, 4, 4]

        # Flatten to [B * k_loss * 16] for CE
        flat_best = best_boards.view(B * k_loss, -1).view(-1) - 1      # [B*k_loss*16] 0-indexed

    # Repeat flat_logits k_loss times along the batch dimension
    # flat_logits is [B*16, 4]; we need [B*k_loss*16, 4]
    flat_logits_rep = flat_logits.view(B, 16, 4)\
                                 .unsqueeze(1)\
                                 .expand(B, k_loss, 16, 4)\
                                 .reshape(B * k_loss * 16, 4)

    return criterion(flat_logits_rep, flat_best)


# ── Stage-I: annealing with population MCMC ───────────────────────────────────

def train_softgk_baseline(
    epoch, model, device, train_loader, optimizer,
    candidate_cache, T, K=100, k_loss=10, sampling_epoch=10
):
    """
    Stage-I training.

    candidate_cache: [N, K, 4, 4]  — population cache across the dataset
    K:               total candidates kept per example
    k_loss:          how many top-scored candidates contribute to the loss
    sampling_epoch:  how often to refresh the population from scratch
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    accumulated_correct_digit  = 0
    accumulated_correct_sample = 0
    accumulated_correct_board  = 0
    total_digit_samples        = 0

    for batch_idx, (images, board_labels, digit_labels, idxs) in enumerate(train_loader):
        images       = images.to(device)
        board_labels = board_labels.to(device)
        digit_labels = digit_labels.to(device)
        B, N, _, C, H, W = images.shape

        flat_images       = images.view(B * N * N, C, H, W)
        digit_flat_boards = digit_labels.view(-1)

        optimizer.zero_grad()
        flat_logits    = model(flat_images)                    # [B*16, 4]
        flat_sq_logits = flat_logits.view(B, N * N, -1)       # [B, 16, 4]

        # ── Update population cache ───────────────────────────────────────
        if epoch == 0:
            with torch.no_grad():
                batch_k_sample = sampler.batch_k_sample(device, board_labels, batch_size=B, k=K)  # [B, K, 4, 4]
                candidate_cache[idxs] = batch_k_sample
        elif (epoch % sampling_epoch == 0) and (epoch != 0):
            # Refresh: draw K fresh valid/invalid candidates
            with torch.no_grad():
                population = candidate_cache[idxs]
                candidate_cache[idxs] = metropolis_walk_population(
                    flat_sq_logits, population, board_labels, T
                )
        else:
            pass

        population = candidate_cache[idxs] # [B, K, 4, 4]
        scores   = compute_score_baseline(flat_sq_logits, population)  # [B, K]

        # ── Loss over top-k_loss candidates ──────────────────────────────
        best    = scores.argmax(dim=1)                        # [B]
        best_boards = population[torch.arange(B, device=device), best]  # [B, 4, 4]
        loss = criterion(flat_logits, best_boards.view(-1) - 1)
        
        #loss = topk_ce_loss(flat_logits, population, flat_sq_logits.detach(), k_loss) 
        loss.backward()
        optimizer.step()

        # ── Metrics (argmax prediction vs ground truth) ───────────────────
        with torch.no_grad():
            pred_flat_digits  = flat_logits.argmax(dim=-1)
            pred_sq_digits    = pred_flat_digits.view(B, 4, 4) + 1
            pred_board_labels = sampler.tensor_check(pred_sq_digits)

            # Best single pseudo-label per example (top-1) for sample accuracy
            scores   = compute_score_baseline(flat_sq_logits.detach(), population)  # [B, K]
            best_idx = scores.argmax(dim=1)                                          # [B]
            best_boards = population[torch.arange(B, device=device), best_idx]      # [B, 4, 4]
            flat_best_digits = best_boards.view(-1)                                  # [B*16]

        total_loss                 += loss.item()
        accumulated_correct_digit  += (pred_flat_digits == (digit_flat_boards - 1)).sum().item()
        accumulated_correct_sample += ((flat_best_digits - 1) == (digit_flat_boards - 1)).sum().item()
        accumulated_correct_board  += (pred_board_labels == board_labels).sum().item()
        total_digit_samples        += digit_flat_boards.size(0)

    n_batches = len(train_loader)
    n_boards  = total_digit_samples / 16
    return (
        total_loss / n_batches,
        accumulated_correct_digit  / total_digit_samples,
        accumulated_correct_board  / n_boards,
        accumulated_correct_sample / total_digit_samples,
    )


# ── Stage-II: backward fine-tuning with top-k loss ────────────────────────────

def train_softgk_backward(model, device, train_loader, optimizer, K=100, k_loss=10):
    """
    Stage-II training.

    Each iteration samples K fresh candidates, scores them, and trains
    on the top-k_loss — no cache, no MCMC, pure greedy K-best.
    """
    model.train()
    total_loss = 0.0
    accumulated_correct_digit  = 0
    accumulated_correct_sample = 0
    accumulated_correct_board  = 0
    total_digit_samples        = 0

    for batch_idx, (images, board_labels, digit_labels, idxs) in enumerate(train_loader):
        images       = images.to(device)
        board_labels = board_labels.to(device)
        digit_labels = digit_labels.to(device)
        B, N, _, C, H, W = images.shape

        flat_images       = images.view(B * N * N, C, H, W)
        digit_flat_boards = digit_labels.view(-1)

        flat_logits    = model(flat_images)                    # [B*16, 4]
        flat_sq_logits = flat_logits.view(B, N * N, -1)       # [B, 16, 4]

        with torch.no_grad():
            population = sampler.batch_k_sample(
                device, board_labels, batch_size=B, k=K
            )                                                  # [B, K, 4, 4]

        optimizer.zero_grad()
        loss = topk_ce_loss(flat_logits, population, flat_sq_logits.detach(), k_loss)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_flat_digits  = flat_logits.argmax(dim=-1)
            pred_sq_digits    = pred_flat_digits.view(B, 4, 4) + 1
            pred_board_labels = sampler.tensor_check(pred_sq_digits)

            scores   = compute_score_baseline(flat_sq_logits.detach(), population)
            best_idx = scores.argmax(dim=1)
            best_boards      = population[torch.arange(B, device=device), best_idx]
            flat_best_digits = best_boards.view(-1)

        total_loss                 += loss.item()
        accumulated_correct_digit  += (pred_flat_digits == (digit_flat_boards - 1)).sum().item()
        accumulated_correct_sample += ((flat_best_digits - 1) == (digit_flat_boards - 1)).sum().item()
        accumulated_correct_board  += (pred_board_labels == board_labels).sum().item()
        total_digit_samples        += digit_flat_boards.size(0)

    n_batches = len(train_loader)
    n_boards  = total_digit_samples / 16
    return (
        total_loss / n_batches,
        accumulated_correct_digit  / total_digit_samples,
        accumulated_correct_board  / n_boards,
        accumulated_correct_sample / total_digit_samples,
    )


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluatek(model, test_loader, device):
    model.eval()
    accumulated_correct_digit = 0
    accumulated_correct_board = 0
    total_digit_samples       = 0

    with torch.no_grad():
        for batch_idx, (images, board_labels, digit_labels, idxs) in enumerate(test_loader):
            images       = images.to(device)
            board_labels = board_labels.to(device)
            digit_labels = digit_labels.to(device)
            B, N, _, C, H, W = images.shape

            flat_images       = images.view(B * N * N, C, H, W)
            digit_flat_boards = digit_labels.view(-1)

            flat_logits      = model(flat_images)
            pred_flat_digits = flat_logits.argmax(dim=-1)
            pred_sq_digits   = pred_flat_digits.view(B, 4, 4) + 1

            pred_board_labels = sampler.tensor_check(pred_sq_digits)

            accumulated_correct_digit += (pred_flat_digits == (digit_flat_boards - 1)).sum().item()
            accumulated_correct_board += (pred_board_labels == board_labels).sum().item()
            total_digit_samples       += digit_flat_boards.size(0)

    return (
        accumulated_correct_digit / total_digit_samples,
        accumulated_correct_board / (total_digit_samples / 16),
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    set_seed(0)

    # Hyperparameters
    T0             = T = 1.0
    t              = 1
    schedule       = 'exp'
    sampling_epoch = 5   # re-sample population every N epochs; walk in between
    K              = 100  # population size per example
    k_loss         = 2    # top-k candidates used for loss per example
    epochs         = 100

    checkpoint_path = 'src/SoftGroundingk/Sudoku4x4/checkpoint/best_stage1_k.pt'
    baseline_checkpoint_path = 'src/models/Sudoku4x4/checkpoint/best_stage1_k.pt'

    train_loader, test_loader, osat_train_loader, osat_test_loader, anchor_digits = \
        get_mnist_sudoku4x4_dataloader(device, n_train=100, n_test=1000, batch_size=64)

    # Cache: [N, K, 4, 4] — full population per training example
    candidate_cache = torch.zeros(
        (len(train_loader.dataset), K, 4, 4), dtype=torch.long #type:ignore
    ).to(device)

    model     = LeNet(num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    T0             = T = 1.0
    t              = 1

    # ── Stage-I  [baseline (k=1)]──────────────────────────────────────────────────────────────
    best_board_acc = 0.0
    for epoch in range(epochs):
        avg_loss, avg_digit_acc, avg_board_acc, avg_sample_acc = train_softgk_baseline(
            epoch, model, device, train_loader, optimizer,
            candidate_cache, T, K=K, k_loss=1, sampling_epoch=sampling_epoch
        )
        test_digit_acc, test_board_acc = evaluatek(model, test_loader, device)

        print(
            f"[Stage-I] (Baseline k_loss=1) Epoch {epoch:3d}: loss={avg_loss:.4f} "
            f"train[digit={avg_digit_acc:.4f} board={avg_board_acc:.4f} sample={avg_sample_acc:.4f}] "
            f"test[digit={test_digit_acc:.4f} board={test_board_acc:.4f}]"
        )

        if test_board_acc > best_board_acc:
            best_board_acc = test_board_acc
            torch.save(model.state_dict(), baseline_checkpoint_path)

        # Temperature annealing (starts after first sampling_epoch)
        if epoch >= sampling_epoch:
            if schedule == 'linear':
                T0 = T0 - 0.05 / math.sqrt(t)
            elif schedule == 'exp':
                T0 = T0 * 0.95
            elif schedule == 'log':
                T0 = T0 / math.log(1 + t)
            t += 1
            T = max(0.01, T0)

    # ── Stage-II ─────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(baseline_checkpoint_path, weights_only=True))
    optimizer_bs = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(40):
        avg_loss, avg_digit_acc, avg_board_acc, avg_sample_acc = train_softgk_backward(
            model, device, train_loader, optimizer_bs, K=K, k_loss=1
        )
        test_digit_acc, test_board_acc = evaluatek(model, test_loader, device)

        print(
            f"[Stage-II] (Baseline k_loss=1) Epoch {epoch:3d}: loss={avg_loss:.4f} "
            f"train[digit={avg_digit_acc:.4f} board={avg_board_acc:.4f} sample={avg_sample_acc:.4f}] "
            f"test[digit={test_digit_acc:.4f} board={test_board_acc:.4f}]"
        )