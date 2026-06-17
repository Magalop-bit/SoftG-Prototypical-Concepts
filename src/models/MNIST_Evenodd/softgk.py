import math
import random
import numpy as np
import torch

from src.backbones.MNIST_Evenodd.lenet import LeNet
from src.databuilder.MNIST_Evenodd.databuilder import get_mnist_addition_dataloader, get_mnist_evenodd_dataloader
from src.samplers.mnistevenodd_sampler import Sampler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def softgk_train_one_epoch(epoch, model, sampler, device, train_loader, optimizer, candidate_cache, T, K=5, sampling_epoch=10):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    accumulated_correct_digits = 0
    accumulated_correct_addition = 0
    accumulated_correct_sample = 0
    total_digit_samples = 0

    for batch_idx, (images, label_add, idxs) in enumerate(train_loader):
        images, label_add = images.to(device), label_add.to(device)
        idxs = idxs.to(device)
        B, N, C, H, W = images.shape

        flat_images = images.view(B * N, C, H, W)  # [B*2, C, H, W], interleaved: d1_0,d2_0,d1_1,d2_1,...

        optimizer.zero_grad()
        logits = model(flat_images)                      # [B*2, 10], interleaved
        logits_d1 = logits[0::2]                         # [B, 10] — every even row is digit-1
        logits_d2 = logits[1::2]                         # [B, 10] — every odd row is digit-2

        if epoch == 0:
            batch_sample_k = sampler.batch_k_sample(label_add, K) # [B, K, 2]
            scores = compute_score_baseline(logits, batch_sample_k) # [B, K]
            candidate_cache[idxs] = batch_sample_k  # [B, K, 2]

        elif (epoch % sampling_epoch == 0) and (epoch != 0):
            batch_sample_k = candidate_cache[idxs]
            walk_batch_sample = metropolis_walk_k(sampler, logits, batch_sample_k, label_add, T)
            candidate_cache[idxs] = walk_batch_sample
        else:
            pass

        sample_digits_k = candidate_cache[idxs]                      # [B, 2]
        scores = compute_score_baseline(logits, sample_digits_k)     #[B, K]
        best = scores.argmax(dim=1)
        sample_digits = sample_digits_k[torch.arange(B, device=device), best]

        #loss = joint_prob_loss(p_1, p_2, sample_digits[:, 0], sample_digits[:, 1])
        loss = criterion(logits, sample_digits.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    return avg_loss

# ── Stage-II: fine-tune with deterministic argmax grounding (backward stage) ──
def train_softgk_backward(model, sampler, device, train_loader, optimizer, K=5):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    accumulated_correct_digits = 0
    accumulated_correct_addition = 0
    accumulated_correct_sample = 0
    total_digit_samples = 0

    for batch_idx, (images, label_add, idxs) in enumerate(train_loader):
        images, label_add = images.to(device), label_add.to(device)
        B, N, C, H, W = images.shape
        flat_images = images.view(B * N, C, H, W)  # [B*2, C, H, W], interleaved: d1_0,d2_0,d1_1,d2_1,...

        optimizer.zero_grad()
        logits = model(flat_images)                      # [B*2, 10], interleaved
        logits_d1 = logits[0::2]                         # [B, 10] — every even row is digit-1
        logits_d2 = logits[1::2]                         # [B, 10] — every odd row is digit-2

        # Backward stage: sample K candidates, pick the highest-scoring one
        with torch.no_grad():
            candidates_k = sampler.batch_k_sample(label_add, K)             # [B, K, 2]
            scores = compute_score_baseline(logits, candidates_k)      # [B, K]
            best = scores.argmax(dim=1)                                      # [B]
            best_digits = candidates_k[torch.arange(B, device=device), best] # [B, 2]

        sample_digits = best_digits
        loss = criterion(logits, sample_digits.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    return avg_loss

# ── Evaluation ──────────────────────────────────────────────────────────────
def evaluatek(model, sampler, device, test_loader, K):
    model.eval()
    accumulated_correct_digits = 0
    accumulated_correct_addition = 0
    total_digit_samples = 0

    with torch.no_grad():
        for images, label_add, label_d1, label_d2, idxs in test_loader:
            images, label_add, label_d1, label_d2 = images.to(device), label_add.to(device), label_d1.to(device), label_d2.to(device)
            B, N, C, H, W = images.shape

            flat_images = images.view(B * N, C, H, W)
            digits = torch.stack([label_d1, label_d2], dim=1)  # [B, 2]

            logits = model(flat_images)                      # [B*2, 10], interleaved
            logits_d1 = logits[0::2]                         # [B, 10] — every even row is digit-1
            logits_d2 = logits[1::2]                         # [B, 10] — every odd row is digit-2

            pred_d1 = logits_d1.argmax(dim=-1)                        # [B]
            pred_d2 = logits_d2.argmax(dim=-1)                        # [B]
            pred_digits = torch.stack([pred_d1, pred_d2], dim=1).view(-1)
            true_digits = digits.view(-1)
            pred_addition = pred_d1 + pred_d2
            true_addition = digits.sum(dim=1)

            accumulated_correct_digits += (pred_digits.view(-1) == true_digits).sum().item()
            accumulated_correct_addition += (pred_addition == true_addition).sum().item()
            total_digit_samples += true_digits.size(0)

    digit_acc = accumulated_correct_digits / total_digit_samples
    addition_acc = accumulated_correct_addition / (total_digit_samples / 2)
    return digit_acc, addition_acc


def metropolis_walk_k(sampler, logits, particles, addition_labels, T):
    """Apply one MH step to each of the K particles independently.
    particles: [B, K, 2]  →  returns [B, K, 2]
    """
    with torch.no_grad():
        B, K, _ = particles.shape
        # Flatten particles to [B*K, 2], replicate labels to match
        flat_particles = particles.reshape(B * K, 2)                         # [B*K, 2]
        rep_labels = addition_labels.repeat_interleave(K)                    # [B*K]

        # Propose one neighbor per particle
        new_flat = sampler.walk(flat_particles.unsqueeze(1), rep_labels).squeeze(1)  # [B*K, 2]

        old_score = compute_score_baseline(logits, particles).reshape(B * K)  # [B*K]
        new_score = compute_score_baseline(logits, new_flat.reshape(B, K, 2)).reshape(B * K)         # [B*K]

        tau = (new_score - old_score) / T
        accept = (torch.log(torch.rand_like(tau)) < tau) | (new_score > old_score)  # [B*K]

        result_flat = torch.where(accept.unsqueeze(-1).expand_as(flat_particles), new_flat, flat_particles)
    return result_flat.reshape(B, K, 2)

def compute_score(logits_d1, logits_d2, digits):
    # logits_d1, logits_d2: [B, 10];  digits: [B, K, 2]
    B, K, _ = digits.shape
    p1 = torch.softmax(logits_d1, dim=-1)[:, None, :].expand(-1, K, -1)  # [B, K, 10]
    p2 = torch.softmax(logits_d2, dim=-1)[:, None, :].expand(-1, K, -1)
    s1 = p1.gather(2, digits[:, :, 0:1]).squeeze(-1)  # [B, K]
    s2 = p2.gather(2, digits[:, :, 1:2]).squeeze(-1)
    return torch.min(s1, s2)                                         # [B, K]

def compute_score_baseline(logits, digits):
    """
    logits: [B*2, 10]
    digits: [B, K, 2]
    """
    B, K, _ = digits.shape

    logits = logits.view(B, 2, 10).unsqueeze(1).expand(-1, K, -1, -1)
    # [B,K,2,10]
    log_probs = torch.log_softmax(logits, dim=-1)

    chosen = log_probs.gather(
        dim=3,
        index=digits.unsqueeze(-1)
    ).squeeze(-1)
    # [B,K,2]

    scores = chosen.sum(dim=-1)   # [B,K]
    return scores



if __name__ == "__main__":
    seed = 0
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler = Sampler()

    T0 = T = 1.0
    t = 1
    schedule = 'exp'
    sampling_epoch = 10
    K = 10  # number of candidates per sample
    epochs = 40
    checkpoint_path = 'src/models/MNIST_Evenodd/checkpoint/best_stage1_k.pt'
    #particle_checkpoint_path = 'src/SoftGroundingK/MNIST_Evenodd/checkpoint/best_particles_stage1.pt'
    #particle_best_checkpoint_path = 'src/SoftGroundingK/MNIST_Evenodd/checkpoint/best_particles_best_stage1.pt'

    train_loader, test_loader, anchor = get_mnist_evenodd_dataloader(n_train=6720, n_test=960, b_size=64)
    #train_loader, test_loader, anchor = get_mnist_addition_dataloader(n_train=6720, n_test=960, b_size=64)
    candidate_cache = torch.zeros((len(train_loader.dataset), K, 2), dtype=torch.long).to(device) #type:ignore

    # ── Stage-I: forward sampling with MCMC + annealing ─────────────────────
    best_addition_acc = 0.0
    for epoch in range(epochs):
        train_loss = softgk_train_one_epoch(
            epoch, model, sampler, device, train_loader, optimizer, candidate_cache, T, K, sampling_epoch=sampling_epoch)

        test_digit_acc, test_addition_acc = evaluatek(model, sampler, device, test_loader, K)

        print(f"[Stage-I] (K Sample -Cache) Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} | "
              f"Test Digit Acc={test_digit_acc:.4f}, Test Addition Acc={test_addition_acc:.4f} ")

        # Save best Stage-I checkpoint based on test addition accuracy
        if test_addition_acc > best_addition_acc:
            best_addition_acc = test_addition_acc
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

    for epoch in range(20):
        train_loss = train_softgk_backward(
            model, sampler, device, train_loader, optimizer_bs, K)

        test_digit_acc, test_addition_acc = evaluatek(model, sampler, device, test_loader, K)

        print(f"[Stage-II] (K Sample -Cache) Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} | "
              f"Test Digit Acc={test_digit_acc:.4f}, Test Addition Acc={test_addition_acc:.4f} ")