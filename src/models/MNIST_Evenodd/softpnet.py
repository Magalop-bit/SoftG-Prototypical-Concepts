import math
import random
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, TensorDataset


from src.samplers.mnistevenodd_sampler import Sampler
from src.backbones.MNIST_Evenodd.protoencoder import ProtoEncoder, compute_prototypes, prototypical_loss, digit_probs, split_support_query, augment_support_set

# ---------------------------------------------------------------------------
# 7.  FULL TRAINING LOOP  (Algorithm 1 from the paper)
# ---------------------------------------------------------------------------

def softpnet_train_one_epoch(
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
    """
    One full epoch:
      - `episodes` prototypical episodes on the augmented support set
      - One pass over the unsupervised loader with joint proto+SL loss
    """
    encoder.train()
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # -----------------------------------------------------------------------
    # PHASE 1: episodic prototypical loss on the augmented support set
    # -----------------------------------------------------------------------
    ep_losses, ep_accs = [], []
    for ep in range(episodes):
        s_img, s_lbl, q_img, q_lbl = split_support_query(aug_images, aug_labels, k_support)
        s_img, s_lbl = s_img.to(device), s_lbl.to(device)
        q_img, q_lbl = q_img.to(device), q_lbl.to(device)

        # Embed support + query together so gradients flow through all
        all_imgs = torch.cat([s_img, q_img], dim=0)   # (S+Q, D)
        all_lbls = torch.cat([s_lbl, q_lbl], dim=0)
        embeddings = encoder(all_imgs)

        optimizer.zero_grad()
        loss, acc = prototypical_loss(embeddings, all_lbls, n_support=k_support)
        loss.backward()
        optimizer.step()

        ep_losses.append(loss.item())
        ep_accs.append(acc.item())

    # -----------------------------------------------------------------------
    # PHASE 2: joint prototypical + Semantic Loss on the unsupervised pairs
    # -----------------------------------------------------------------------
    # Re-compute fixed prototypes from the clean (1-per-class) support set.
    # These prototypes are recomputed each time to ensure they stay aligned
    # with the current encoder state (gradients flow through them too).
    nesy_losses, nesy_sl, nesy_proto = [], [], []
    warmup_epoch = 0
    for (images, y_sum, idx) in unsup_loader:
        img1, img2 = images[:,0], images[:,1]
        img1, img2, y_sum = img1.to(device), img2.to(device), y_sum.to(device)
        idx = idx.to(device)

        optimizer.zero_grad()
        B = images.size(0)

        # --- prototypes from clean support set (gradients flow through) ---
        prototypes = compute_prototypes(proto_images, proto_labels, encoder)

        # --- embed both images in the pair ---
        emb1 = encoder(img1)                        # (B, D)
        emb2 = encoder(img2)                        # (B, D)

        # --- prototypical classification probabilities ---
        p1 = digit_probs(emb1, prototypes)          # (B, 10)
        p2 = digit_probs(emb2, prototypes)          # (B, 10)

        logits_1    = -(torch.cdist(emb1, prototypes, p=2) ** 2)  # (Q, n_classes)
        logits_2    = -(torch.cdist(emb2, prototypes, p=2) ** 2)  # (Q, n_classes)
        logits      = torch.stack([logits_1, logits_2], dim=1)    # (B, 2, 10)
        
        if epoch == 0:
            with torch.no_grad():
                batch_sample_k = sampler.batch_k_sample(y_sum, K*2) # [B, K*2, 2]
                scores = compute_score_baseline(logits, batch_sample_k) # [B, K*2]
                best_values, best = torch.topk(scores, K, dim=1)
                candidate_cache[idx] = batch_sample_k[torch.arange(B, device=device).unsqueeze(1), best]  # [B, K, 2]

        elif (epoch % sampling_epoch == 0) and (epoch != 0):
            with torch.no_grad():
                batch_sample_k = candidate_cache[idx]
                walk_batch_sample = metropolis_walk_k(sampler, logits, batch_sample_k, y_sum, T)
                candidate_cache[idx] = walk_batch_sample
        else:
            pass

        candidates = candidate_cache[idx]                      # [B, 2]
        scores = compute_score_baseline(logits, candidates)     #[B, K]
        best = scores.argmax(dim=1)
        candidate_digits = candidates[torch.arange(B, device=device), best]

        loss_ce  = criterion(
            logits.reshape(-1, 10),
            candidate_digits.reshape(-1)
        )

        # --- compute score per sample ---


        # --- Semantic Loss ---
        #loss_sl = semantic_loss_mnist(p1, p2, y_sum)

        # --- Episodic prototypical loss on a fresh mini-episode drawn from
        #     the augmented support set (one per NeSy batch, as in Algorithm 1
        #     where both losses appear in the same loop iteration).           ---
        s_img, s_lbl, q_img, q_lbl = split_support_query(aug_images, aug_labels, k_support)
        s_img, s_lbl = s_img.to(device), s_lbl.to(device)
        q_img, q_lbl = q_img.to(device), q_lbl.to(device)
        ep_emb = encoder(torch.cat([s_img, q_img]))
        ep_lbl = torch.cat([s_lbl, q_lbl])
        loss_proto, _ = prototypical_loss(ep_emb, ep_lbl, n_support=k_support)


        loss = sl_weight * loss_ce + loss_proto
        
        loss.backward()
        optimizer.step()

        nesy_losses.append(loss.item())
        nesy_sl.append(loss_ce.item())
        nesy_proto.append(loss_proto.item())

    return {
        "ep_loss":    sum(ep_losses)   / max(len(ep_losses), 1),
        "ep_acc":     sum(ep_accs)     / max(len(ep_accs),   1),
        "nesy_loss":  sum(nesy_losses) / max(len(nesy_losses),1),
        "nesy_sl":    sum(nesy_sl)     / max(len(nesy_sl),    1),
        "nesy_proto": sum(nesy_proto)  / max(len(nesy_proto), 1),
    }

@torch.no_grad()
def evaluate(
    encoder:      ProtoEncoder,
    test_loader:  DataLoader,
    proto_images: torch.Tensor,
    proto_labels: torch.Tensor,
    device:       torch.device,
) -> dict:
    """
    Evaluates:
      - digit_acc   – concept accuracy: fraction of individual digits correct
                      (mirrors Acc(C) from Table 1 of the paper)
      - digit_f1    – macro-averaged F1 over the 10 digit classes (F1(C))
      - sum_acc     – label accuracy: fraction of sums correct (Acc(Y))
    """
    encoder.eval()
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)
    prototypes   = compute_prototypes(proto_images, proto_labels, encoder)

    # Accumulators for per-class TP / FP / FN (for macro F1)
    tp = torch.zeros(10)
    fp = torch.zeros(10)
    fn = torch.zeros(10)

    correct_digit, correct_sum, total_digits, total_pairs = 0, 0, 0, 0

    # Test loader yields (img1, img2, true_d1, true_d2, y_sum)
    for (images, y_sum, true_d1, true_d2, _) in test_loader:
        img1, img2 = images[:,0], images[:,1]
        img1    = img1.to(device)
        img2    = img2.to(device)
        true_d1 = true_d1.to(device)
        true_d2 = true_d2.to(device)
        y_sum   = y_sum.to(device)

        p1 = digit_probs(encoder(img1), prototypes)   # (B, 10)
        p2 = digit_probs(encoder(img2), prototypes)   # (B, 10)

        pred_d1  = p1.argmax(1)                        # (B,)
        pred_d2  = p2.argmax(1)
        pred_sum = pred_d1 + pred_d2

        # --- label accuracy ---
        correct_sum  += (pred_sum == y_sum).sum().item()
        total_pairs  += y_sum.size(0)

        # --- concept accuracy (both digits in the pair) ---
        correct_digit += (pred_d1 == true_d1).sum().item()
        correct_digit += (pred_d2 == true_d2).sum().item()
        total_digits  += 2 * y_sum.size(0)

        # --- per-class TP / FP / FN for macro F1 ---
        for d in range(10):
            tp[d] += ((pred_d1 == d) & (true_d1 == d)).sum().item()
            tp[d] += ((pred_d2 == d) & (true_d2 == d)).sum().item()
            fp[d] += ((pred_d1 == d) & (true_d1 != d)).sum().item()
            fp[d] += ((pred_d2 == d) & (true_d2 != d)).sum().item()
            fn[d] += ((pred_d1 != d) & (true_d1 == d)).sum().item()
            fn[d] += ((pred_d2 != d) & (true_d2 == d)).sum().item()

    # Macro F1: average per-class F1, with 0 when a class has no support
    precision = tp / (tp + fp).clamp(min=1e-8)
    recall    = tp / (tp + fn).clamp(min=1e-8)
    f1_per    = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    macro_f1  = f1_per.mean().item()

    return {
        "digit_acc": 100.0 * correct_digit / max(total_digits, 1),
        "digit_f1":  100.0 * macro_f1,
        "sum_acc":   100.0 * correct_sum   / max(total_pairs,  1),
    }

def metropolis_walk_baseline(sampler, logits, digits, addition_labels, T):
    with torch.no_grad():
        old_digits = digits.unsqueeze(1).clone()
        new_digits = sampler.walk(old_digits, addition_labels).to(logits.device)  # [B, K, 2]

        old_P = compute_score_baseline(logits, old_digits)
        new_P = compute_score_baseline(logits, new_digits)

        if T != 0.0:
            tau = (new_P - old_P) / T
            accept = (torch.log(torch.rand_like(tau)) < tau) #| (new_P > old_P)
        else:
            accept = (new_P > old_P)

        res = torch.where(accept.unsqueeze(-1).expand_as(new_digits), new_digits, old_digits)

    return res  # [B, K, 2]      

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

        if T != 0.0:
            tau = (new_score - old_score) / T
            accept = (torch.log(torch.rand_like(tau)) < tau) #| (new_score > old_score)  # [B*K]
        else:
            accept = (new_score > old_score)

        result_flat = torch.where(accept.unsqueeze(-1).expand_as(flat_particles), new_flat, flat_particles)
    return result_flat.reshape(B, K, 2)

def compute_score_baseline(logits, digits):
    """
    logits: [B, 2, 10]
    digits: [B, K, 2]
    """
    digits = digits.to(logits.device)
    B, K, _ = digits.shape

    logits = logits.unsqueeze(1).expand(-1, K, -1, -1)
    # [B,K,2,10]
    log_probs = torch.log_softmax(logits, dim=-1)

    chosen = log_probs.gather(
        dim=3,
        index=digits.unsqueeze(-1)
    ).squeeze(-1)
    # [B,K,2]

    scores = chosen.sum(dim=-1)   # [B,K]
    return scores

# ---------------------------------------------------------------------------
# 10.  MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sampler = Sampler()

    from src.databuilder.MNIST_Evenodd.databuilder import get_mnist_evenodd_dataloader
    # --- Data ---
    train_loader, test_loader, proto_images = get_mnist_evenodd_dataloader(
        n_train=6720, n_test=960, b_size=64
    )
    proto_images = proto_images.to(device)
    proto_labels = torch.arange(len(proto_images)).to(device)

    # --- Augment the 1-per-class support set ---
    aug_images, aug_labels = augment_support_set(
        proto_images, proto_labels, device, num_augmentations=9
    )

    # aug_images: (10 * 10, 1, 28, 28) = (100, 1, 28, 28)
    print(f"Augmented support set: {aug_images.shape[0]} images, "
          f"{torch.unique(aug_labels).numel()} classes")

    # --- Model & optimizer ---
    encoder   = ProtoEncoder(x_dim=1, hid_dim=64, z_dim=64).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    T0 = T = 1.0
    t = 1
    K = 20
    schedule = 'exp'
    sampling_epoch = 1

    candidate_cache = torch.zeros((len(train_loader.dataset), K, 2), dtype=torch.long).to(device) #type:ignore
    checkpoint_path = "src/models/MNIST_Evenodd/checkpoints/best_encoder_evenodd.pth"
    checkpoint_path_bs = "src/models/MNIST_Evenodd/checkpoints/best_encoder_evenodd_bs.pth"

    # --- Training ---
    num_epochs = 10       # paper uses 10 epochs for MNIST-EvenOdd
    sl_weight  = 5.0     #10.0     # paper uses wsl = 10.0
    episodes   = 50       # episodic iterations per epoch

    best_sum_acc = 0.0
    for epoch in range(0, num_epochs):
        stats = softpnet_train_one_epoch(
            encoder      = encoder,
            optimizer    = optimizer,
            aug_images   = aug_images,
            aug_labels   = aug_labels,
            unsup_loader = train_loader,
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

        eval_stats = evaluate(encoder, test_loader, proto_images, proto_labels, device)

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"ep_loss={stats['ep_loss']:.4f} ep_acc={stats['ep_acc']:.4f} | "
            f"nesy_loss={stats['nesy_loss']:.4f} "
            f"(sl={stats['nesy_sl']:.4f} proto={stats['nesy_proto']:.4f}) | "
            f"digit_acc={eval_stats['digit_acc']:.1f}% "
            f"digit_f1={eval_stats['digit_f1']:.1f}% "
            f"sum_acc={eval_stats['sum_acc']:.1f}%"
        )

        if eval_stats["sum_acc"] > best_sum_acc:
            best_sum_acc = eval_stats["sum_acc"]
            os.makedirs("src/models/MNIST_Evenodd/checkpoints", exist_ok=True)
            torch.save(encoder.state_dict(), checkpoint_path)
            print(f"  → Saved best model (sum acc = {best_sum_acc:.1f}%, "
                  f"digit acc = {eval_stats['digit_acc']:.1f}%, "
                  f"F1(C) = {eval_stats['digit_f1']:.1f}%)")
            
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
            

    print(f"\nTraining complete. Best sum accuracy: {best_sum_acc:.1f}%")