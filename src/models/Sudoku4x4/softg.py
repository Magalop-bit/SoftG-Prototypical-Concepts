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

def joint_prob_loss(logits_board, boards):
    """
    logits_board: [B,16,4]
    boards:       [B,4,4]
    """

    B = boards.shape[0]

    flat_boards = boards.view(B, -1) - 1

    log_probs = torch.log_softmax(logits_board, dim=-1)

    chosen_log_probs = log_probs.gather(
        dim=2,
        index=flat_boards.unsqueeze(-1)
    ).squeeze(-1)  # [B,16]

    # joint log probability
    joint_log_prob = chosen_log_probs.sum(dim=-1)

    return -joint_log_prob.mean()


def train_softg_baseline(epoch, model, device, train_loader, optimizer, candidate_cache, T, K=5, sampling_epoch=10):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    accumulated_correct_digit = 0
    accumulated_correct_sample = 0
    accumulated_correct_board = 0
    total_digit_samples = 0
    
    for batch_idx, (images, board_labels, digit_labels, idxs) in enumerate(train_loader):
        images       = images.to(device)
        board_labels = board_labels.to(device)
        digit_labels = digit_labels.to(device)
        B, N, _, C, H, W = images.shape

        flat_images = images.view(B * N * N, C, H, W)
        digit_flat_boards = digit_labels.view(-1)

        # ------------------------------------------------------------------
        # Single forward pass — reuse logits everywhere
        # ------------------------------------------------------------------
        optimizer.zero_grad()
        flat_logits    = model(flat_images)                    # [B*16, 4]
        flat_sq_logits = flat_logits.view(B, N * N, -1)       # [B, 16, 4]

        # ------------------------------------------------------------------
        # Update pseudo-label cache
        # ------------------------------------------------------------------
        if epoch % sampling_epoch == 0:
            batch_sample_k = sampler.batch_k_sample(device, board_labels, batch_size=B, k=K)
            scores = compute_score_baseline(flat_sq_logits.detach(), batch_sample_k)
            best   = scores.argmax(dim=1)
            candidate_cache[idxs] = batch_sample_k[torch.arange(B, device=device), best]
        else:
            batch_sample     = candidate_cache[idxs]
            walk_batch_sample = metropolis_walk(flat_sq_logits.detach(), batch_sample, board_labels, T)
            candidate_cache[idxs] = walk_batch_sample.squeeze(1)

        sample_digits     = candidate_cache[idxs]              # [B, 4, 4]
        flat_sample_digits = sample_digits.view(-1)            # [B*16]

        # ------------------------------------------------------------------
        # Loss: CE on pseudo-labels + distribution-level constraint loss
        # ------------------------------------------------------------------
        ce_loss   = criterion(flat_logits, flat_sample_digits - 1)
        #cons_loss = joint_prob_loss(flat_sq_logits, sample_digits)
        loss      = ce_loss #ce_loss + cons_loss
        loss.backward()
        optimizer.step()

        pred_flat_digits = flat_logits.argmax(dim=-1)
        pred_sq_digits = pred_flat_digits.view(B, 4, 4) + 1
        pred_board_labels = sampler.tensor_check(pred_sq_digits)


        total_loss += loss.item()
        accumulated_correct_digit  += (pred_flat_digits == (digit_flat_boards - 1)).sum().item()
        accumulated_correct_sample += ((flat_sample_digits - 1)   == (digit_flat_boards - 1)).sum().item()
        accumulated_correct_board += (pred_board_labels == board_labels).sum().item()
        total_digit_samples        += digit_flat_boards.size(0)

    avg_loss           = total_loss / len(train_loader)
    avg_digit_accuracy = accumulated_correct_digit  / total_digit_samples
    avg_sample_accuracy = accumulated_correct_sample / total_digit_samples
    avg_board_accuracy = accumulated_correct_board / (total_digit_samples/16)

    return avg_loss, avg_digit_accuracy, avg_board_accuracy, avg_sample_accuracy

def evaluate(model, test_loader, device, K):
    """
    Evaluate the model on the test set.
    
    Returns:
        digit_acc: per-digit accuracy against ground truth
        board_acc: per-board SAT/UNSAT classification accuracy
    """
    model.eval()
    accumulated_correct_digit = 0
    accumulated_correct_board = 0
    total_digit_samples = 0

    with torch.no_grad():
        for batch_idx, (images, board_labels, digit_labels, idxs) in enumerate(test_loader):
            images       = images.to(device)
            board_labels = board_labels.to(device)
            digit_labels = digit_labels.to(device)
            B, N, _, C, H, W = images.shape

            flat_images = images.view(B * N * N, C, H, W)
            digit_flat_boards = digit_labels.view(-1)

            # Forward pass
            flat_logits    = model(flat_images)                    # [B*16, 4]
            flat_sq_logits = flat_logits.view(B, N * N, -1)       # [B, 16, 4]

            # Direct prediction (argmax)
            pred_flat_digits = flat_sq_logits.argmax(dim=-1).view(-1) #            # [B*16]
            pred_sq_digits = pred_flat_digits.view(B, 4, 4) + 1    # [B, 4, 4], values 1..4
            
            # Board-level validity check
            pred_board_labels = sampler.tensor_check(pred_sq_digits)  # [B] bool
            
            # Metrics
            accumulated_correct_digit += (pred_flat_digits == (digit_flat_boards-1)).sum().item()
            accumulated_correct_board += (pred_board_labels == board_labels).sum().item()
            total_digit_samples += digit_flat_boards.size(0)

    avg_digit_accuracy = accumulated_correct_digit / total_digit_samples
    avg_board_accuracy = accumulated_correct_board / (total_digit_samples / 16)

    return avg_digit_accuracy, avg_board_accuracy

# ── Stage-II: fine-tune with deterministic argmax grounding (backward stage) ──
def train_softg_backward(model, device, train_loader, optimizer, K=5):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    accumulated_correct_digit = 0
    accumulated_correct_sample = 0
    accumulated_correct_board = 0
    total_digit_samples = 0

    for batch_idx, (images, board_labels, digit_labels, idxs) in enumerate(train_loader):
        images       = images.to(device)
        board_labels = board_labels.to(device)
        digit_labels = digit_labels.to(device)
        B, N, _, C, H, W = images.shape

        flat_images = images.view(B * N * N, C, H, W)  # [B * 4 * 4, C, H, W]
        digit_flat_boards = digit_labels.view(-1)

        # Forward pass
        flat_logits    = model(flat_images)                    # [B*16, 4]
        flat_sq_logits = flat_logits.view(B, N * N, -1)       # [B, 16, 4]

        # Backward stage: sample K candidates, pick the highest-scoring one
        with torch.no_grad():
            batch_sample_k = sampler.batch_k_sample(device, board_labels, batch_size=B, k=K)# [B, K, 4, 4]
            scores = compute_score_baseline(flat_sq_logits.detach(), batch_sample_k)# [B, K]
            best = scores.argmax(dim=1)                                             # [B]
            best_digits = batch_sample_k[torch.arange(B, device=device), best]      # [B, 4, 4]

        optimizer.zero_grad()
        flat_sample_digits = best_digits.view(-1)
        loss   = criterion(flat_logits, flat_sample_digits - 1)
        #loss = joint_prob_loss(flat_sq_logits, sample_digits)
        loss.backward()
        optimizer.step()

        pred_flat_digits = flat_logits.argmax(dim=-1)
        pred_sq_digits = pred_flat_digits.view(B, 4, 4) + 1
        pred_board_labels = sampler.tensor_check(pred_sq_digits)


        total_loss += loss.item()
        accumulated_correct_digit  += (pred_flat_digits == (digit_flat_boards - 1)).sum().item()
        accumulated_correct_sample += ((flat_sample_digits - 1)   == (digit_flat_boards - 1)).sum().item()
        accumulated_correct_board += (pred_board_labels == board_labels).sum().item()
        total_digit_samples        += digit_flat_boards.size(0)

    avg_loss           = total_loss / len(train_loader)
    avg_digit_accuracy = accumulated_correct_digit  / total_digit_samples
    avg_sample_accuracy = accumulated_correct_sample / total_digit_samples
    avg_board_accuracy = accumulated_correct_board / (total_digit_samples/16)

    return avg_loss, avg_digit_accuracy, avg_board_accuracy, avg_sample_accuracy

def compute_score_baseline(logits, boards):
    """
    logits: [B, 16, 4]
    boards: [B, K, 4, 4]
    """
    B, K, _, _ = boards.shape

    logits = logits.unsqueeze(1).expand(-1, K, -1, -1)
    # [B,K,16,4]
    log_probs = torch.log_softmax(logits, dim=-1)
    flat_boards = boards.clone().view(B, K, -1) - 1 #[B, K, 16]

    chosen = log_probs.gather(
        dim=3,
        index=flat_boards.unsqueeze(-1)
    ).squeeze(-1)
    # [B,K,16]

    scores = chosen.sum(dim=-1)   # [B,K]
    return scores

def metropolis_walk(logits_board, digits_board, board_labels, T):
    with torch.no_grad():
        old_boards = digits_board.clone().unsqueeze(1) # [B, 1, 4, 4]
        new_boards = sampler.walk(digits_board, board_labels).unsqueeze(1)  # [B, 1, 4, 4]

        old_P = compute_score_baseline(logits_board, old_boards)
        new_P = compute_score_baseline(logits_board, new_boards)

        tau = (new_P - old_P) / T
        accept = (torch.log(torch.rand_like(tau)) < tau) #| (new_P > old_P)

        res = torch.where(accept.unsqueeze(-1).unsqueeze(-1).expand_as(new_boards), new_boards, old_boards)

    return res  # [B, K, 4, 4]

if __name__ == '__main__':
    T0 = T = 1.0
    t = 1
    schedule = 'exp'
    sampling_epoch = 10
    K = 100  # number of candidates per sample
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = 'src/models/Sudoku4x4/checkpoint/best_stage1.pt'

    train_loader, test_loader, osat_train_loader, osat_test_loader, anchor_digits = get_mnist_sudoku4x4_dataloader(device, n_train=100, n_test=1000, batch_size=64)
    candidate_cache = torch.zeros((len(train_loader.dataset), 4, 4), dtype=torch.long).to(device) #type:ignore

    model = LeNet(num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    # ── Stage-I: forward sampling with MCMC + annealing ─────────────────────
    best_addition_acc = 0.0
    for epoch in range(epochs):
        avg_loss, avg_digit_acc, avg_board_acc, avg_sample_acc = train_softg_baseline(epoch, model, device, train_loader, optimizer, candidate_cache, T, K=K, sampling_epoch=sampling_epoch)
        test_digit_acc, test_board_acc = evaluate(model, test_loader, device, K=300)
        
        print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, Train Digit Accuracy: {avg_digit_acc:.4f}, Train Board Accuracy: {avg_board_acc:.4f}, Train Sample Accuracy: {avg_sample_acc:.4f} | "
              f"Test Digit: {test_digit_acc:.4f}, Test Board: {test_board_acc:.4f}"
              )

        # Save best Stage-I checkpoint based on test addition accuracy
        if test_board_acc > best_addition_acc:
            best_addition_acc = test_board_acc
            torch.save(model.state_dict(), checkpoint_path)

        if epoch >= sampling_epoch:
            if schedule == 'linear':
                dT = 0.05 * 1.0 / math.sqrt(t)
                T0 = T0 - dT
            elif schedule == 'exp':
                T0 = T0 * 0.95
            elif schedule == 'log':
                T0 = T0 / math.log(1 + t)
            t += 1
            T = max(0.01, T0)

    # ── Stage-II: backward fine-tuning from best Stage-I checkpoint ──────────
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    optimizer_bs = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(40):
        avg_loss, avg_digit_acc, avg_board_acc, avg_sample_acc = train_softg_backward(model, device, train_loader, optimizer_bs, K=K)
        test_digit_acc, test_board_acc = evaluate(model, test_loader, device, K=300)

        print(f"[Stage-II] Epoch {epoch}: "
              f"Train Loss={avg_loss:.4f}, "
              f"Train Digit Acc={avg_digit_acc:.4f}, Train Board Acc={avg_board_acc:.4f}, Train Sample Acc={avg_sample_acc:.4f} | "
              f"Test Digit Acc={test_digit_acc:.4f}, Test Board Acc={test_board_acc:.4f} ")