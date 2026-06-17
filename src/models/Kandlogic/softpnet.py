import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader

import random
import numpy as np

from src.models.Kandlogic.helpers.kandlogic import kand_label, _row_predicate, augment_support_set, split_support_query
from src.backbones.Kandlogic.protoencoder import (PrimitivesProtoNet, prototypical_loss, digit_probs, compute_prototypes)

from src.samplers.kandlogic_sampler import Sampler

# ── Kandinsky constants ────────────────────────────────────────────────────
figures = {0: 'square', 1: 'circle', 2: 'triangle'}
colors  = {0: 'red',    1: 'yellow', 2: 'blue'}

N            = 3   # scenes per scenario
OBJECTS      = 3   # objects per scene
BOARD_CELLS  = N * OBJECTS          # 9 cells
NUM_PRIMITIVES = N * OBJECTS        # 9 classes: shape ∈ {0,1,2} × color ∈ {0,1,2}
# primitive_id = shape * 3 + color  ∈ [0, 8]

_PRIMITIVE_LABELS = {
    0: 'red-square',    1: 'red-circle',    2: 'red-triangle',
    3: 'yellow-square', 4: 'yellow-circle', 5: 'yellow-triangle',
    6: 'blue-square',   7: 'blue-circle',   8: 'blue-triangle',
}

def scenario_to_primitive_ids(population: torch.Tensor) -> torch.Tensor:
    """
    [B, K, 3, 3, 2] → [B, K, 9]  primitive IDs in [0, 8].
    primitive_id = shape * 3 + color.
    """
    shapes = population[..., 0]          # [B, K, 3, 3]
    colors = population[..., 1]          # [B, K, 3, 3]
    prim   = shapes * 3 + colors         # [B, K, 3, 3]
    return prim.view(*population.shape[:2], -1)   # [B, K, 9]


def compute_score_kand(
    logits:     torch.Tensor,   # [B, 9, 9]  per-cell logits over 9 primitive classes
    population: torch.Tensor,   # [B, K, 3, 3, 2]  candidate scenarios
) -> torch.Tensor:              # [B, K]
    """
    Joint log-probability score for each candidate scenario.
    Replaces the Sudoku-specific compute_score_baseline.
    """
    B, K = population.shape[:2]
    prim_ids      = scenario_to_primitive_ids(population)          # [B, K, 9]
    log_probs     = torch.log_softmax(logits, dim=-1)              # [B, 9, 9]
    log_probs_exp = log_probs.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, 9, 9]
    chosen = log_probs_exp.gather(
        dim=3,
        index=prim_ids.unsqueeze(-1),                              # [B, K, 9, 1]
    ).squeeze(-1)                                                  # [B, K, 9]
    return chosen.sum(dim=-1)


# ── Metropolis walk ────────────────────────────────────────────────────────

def metropolis_walk_population(
    sampler:      Sampler,
    logits:       torch.Tensor,   # [B, 9, 9]  detached model logits
    population:   torch.Tensor,   # [B, K, 3, 3, 2]
    board_labels: torch.Tensor,   # [B]  1=SAT, 0=UNSAT
    T:            float,
    n_steps:      int = 1,
) -> torch.Tensor:                # [B, K, 3, 3, 2]
    """
    One Metropolis step per candidate in the population.

    Proposes a new scenario via sampler.walk (which keeps SAT→SAT, UNSAT→UNSAT),
    then accepts/rejects based on the log-score ratio.
    """
    with torch.no_grad():
        B, K = population.shape[:2]
        device = population.device

        # Flatten to [B*K, 3, 3, 2] so sampler.walk handles a flat batch
        flat_pop    = population.reshape(B * K, N, OBJECTS, 2)
        # Tile labels: [B] → [B*K]
        flat_labels = board_labels.unsqueeze(1).expand(B, K).reshape(B * K)

        # One walk step per candidate  →  [B*K, 3, 3, 2]
        flat_proposed = sampler.walk(flat_pop, flat_labels, n_steps=n_steps, fixed_pos=1)
        proposed = flat_proposed.view(B, K, N, OBJECTS, 2)          # [B, K, 3, 3, 2]

        # Score old and proposed
        old_scores = compute_score_kand(logits, population)          # [B, K]
        new_scores = compute_score_kand(logits, proposed)            # [B, K]

        # Metropolis acceptance: accept if log u < (new - old) / T
        if T != 0.0:
            log_u  = torch.log(torch.rand_like(old_scores))
            accept = (log_u < (new_scores - old_scores) / T) | (new_scores > old_scores )    # [B, K] bool
        else:
            accept = new_scores > old_scores                         # greedy

        accept_exp = accept.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,K,1,1,1]
        return torch.where(accept_exp.expand_as(proposed), proposed, population)
    


# ── Training loop ─────────────────────────────────────────────────────────

def softpnet_train_one_epoch(
    encoder:         PrimitivesProtoNet,
    optimizer:       torch.optim.Optimizer,
    aug_images:      torch.Tensor,
    aug_labels:      torch.Tensor,
    unsup_loader:    DataLoader,
    proto_images:    torch.Tensor,
    proto_labels:    torch.Tensor,
    device:          torch.device,
    sampler:         Sampler,
    candidate_cache: torch.Tensor,    # [dataset_size, K, 3, 3, 2]  long, on device
    K:               int   = 5,
    sampling_epoch:  int   = 1,
    T:               float = 1.0,
    episodes:        int   = 50,
    k_support:       int   = 1,
    sl_weight:       float = 10.0,
    epoch:           int   = 0,
    n_walk_steps:    int   = 1,
) -> dict:

    sl_warmup = 2
    sl_ramp = 5

    encoder.train()
    aug_images   = aug_images.to(device)
    aug_labels   = aug_labels.to(device)
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)

    if epoch < sl_warmup:
        effective_sl = 0.0
    else:
        effective_sl = sl_weight * min(1.0, (epoch - sl_warmup) / max(sl_ramp, 1))

    # ── Phase 1: episodic prototypical loss ───────────────────────────────
    ep_losses, ep_accs = [], []

    for ep in range(episodes):
        s_img, s_lbl, q_img, q_lbl = split_support_query(aug_images, aug_labels, k_support)
        if q_img.shape[0] == 0:
            continue

        all_emb = encoder(torch.cat([s_img, q_img], dim=0))
        all_lbl = torch.cat([s_lbl, q_lbl], dim=0)

        optimizer.zero_grad()
        loss, acc = prototypical_loss(all_emb, all_lbl, n_support=k_support)
        loss.backward()
        optimizer.step()

        ep_losses.append(loss.item())
        ep_accs.append(acc.item())

    # ── Phase 2: NeSy / Soft-grounding KL loss ───────────────────────────
    nesy_losses, nesy_kl, nesy_proto = [], [], []

    flat_proto_labels = proto_labels[:, 0] * 3 + proto_labels[:, 1]  # primitive ids

    encoder.eval()
    with torch.no_grad():
        prototypes_epoch = compute_prototypes(proto_images, flat_proto_labels, encoder)
    encoder.train()

    for images, labels, _, idx in unsup_loader:
        images   = images.to(device)
        y_scene  = labels.float().to(device)    # [B]  1=SAT, 0=UNSAT
        idx      = idx.to(device)
        B        = y_scene.shape[0]

        # Embed all 9 cells of all B scenarios: [B, 3, 3, C, H, W]
        # images shape from dataloader: [B, N, N, C, H, W]
        _, N_, OBJS_, C, H, W = images.shape
        cells_flat = images.reshape(B * BOARD_CELLS, C, H, W)   # [B*9, C, H, W]
        embs_flat  = encoder(cells_flat)                         # [B*9, D]

        # Logits: negative squared distance to each prototype  [B*9, 9]
        logits_flat     = -(torch.cdist(embs_flat, prototypes_epoch, p=2) ** 2)
        flat_sq_logits  = logits_flat.view(B, BOARD_CELLS, NUM_PRIMITIVES)  # [B, 9, 9]

        optimizer.zero_grad()

        # ── Update candidate cache ────────────────────────────────────────
        with torch.no_grad():
            if (epoch == 0) or (epoch == sl_warmup):
                # Cold start: sample K*10 candidates, keep top-K by score
                batch_k_sample = sampler.batch_k_sample(
                    y_scene, batch_size=B, k=K * 1000
                )                                               # [B, K*10, 3, 3, 2]
                scores     = compute_score_kand(flat_sq_logits.detach(), batch_k_sample)
                _, best    = torch.topk(scores, K, dim=1)      # [B, K]
                candidate_cache[idx] = batch_k_sample[
                    torch.arange(B, device=device).unsqueeze(1), best
                ]                                               # [B, K, 3, 3, 2]

            elif (epoch % sampling_epoch == 0):
                # Metropolis refresh: walk each candidate one step
                population = candidate_cache[idx]               # [B, K, 3, 3, 2]
                candidate_cache[idx] = metropolis_walk_population(
                    sampler,
                    flat_sq_logits.detach(),
                    population,
                    y_scene.long(),
                    T,
                    n_steps=n_walk_steps,
                )

        # ── Soft targets via weighted marginals ───────────────────────────
        population = candidate_cache[idx]                       # [B, K, 3, 3, 2]

        with torch.no_grad():
            raw_scores   = compute_score_kand(flat_sq_logits.detach(), population)  # [B, K]
            weights      = torch.softmax(raw_scores /  max(T, 0.3), dim=1)                     # [B, K]

            # prim_ids[b, k, cell] ∈ [0, 8]
            prim_ids     = scenario_to_primitive_ids(population)  # [B, K, 9]

            # soft_targets[b, cell, prim] = Σ_k w_k * 1[prim_ids[b,k,cell] == prim]
            soft_targets = torch.zeros(B, BOARD_CELLS, NUM_PRIMITIVES, device=device)
            for p in range(NUM_PRIMITIVES):
                soft_targets[:, :, p] = (
                    weights.unsqueeze(-1) * (prim_ids == p).float()
                ).sum(dim=1)                                    # [B, 9]

        # ── KL divergence loss ────────────────────────────────────────────
        log_probs = F.log_softmax(flat_sq_logits, dim=-1)      # [B, 9, 9]
        loss_kl   = F.kl_div(log_probs, soft_targets, reduction='batchmean')

        # ── Episodic proto loss (keeps proto training alive) ──────────────
        s_img, s_lbl, q_img, q_lbl = split_support_query(aug_images, aug_labels, k_support)
        if q_img.shape[0] > 0:
            ep_emb = encoder(torch.cat([s_img, q_img]))
            ep_lbl = torch.cat([s_lbl, q_lbl])
            loss_proto, _ = prototypical_loss(ep_emb, ep_lbl, n_support=k_support)
        else:
            loss_proto = torch.tensor(0.0, device=device)

        loss = loss_proto + effective_sl * loss_kl
        loss.backward()
        optimizer.step()

        nesy_losses.append(loss.item())
        nesy_kl.append(loss_kl.item())
        nesy_proto.append(loss_proto.item())

    return {
        "ep_loss":    sum(ep_losses)   / max(len(ep_losses),   1),
        "ep_acc":     sum(ep_accs)     / max(len(ep_accs),     1),
        "nesy_loss":  sum(nesy_losses) / max(len(nesy_losses), 1),
        "nesy_kl":    sum(nesy_kl)     / max(len(nesy_kl),     1),
        "nesy_proto": sum(nesy_proto)  / max(len(nesy_proto),  1),
    }


# ── Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    encoder:      PrimitivesProtoNet,
    sampler:      Sampler,
    test_loader:  DataLoader,
    proto_images: torch.Tensor,
    proto_labels: torch.Tensor,
    device:       torch.device,
) -> dict:
    """
    concept_acc  — per-primitive classification accuracy (shape + color separately)
    label_acc    — scene-level Kand label accuracy
    label_f1     — macro F1 on the binary scene label
    """
    encoder.eval()
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)

    flat_proto_labels = proto_labels[:, 0] * 3 + proto_labels[:, 1]
    prototypes        = compute_prototypes(proto_images, flat_proto_labels, encoder)

    correct_concept = 0
    total_concept   = 0

    correct_scene = 0
    total_scene   = 0
    tp_k = fp_k = fn_k = tn_k = 0

    for images, labels, concepts, idxs in test_loader:
        images   = images.to(device)
        concepts = concepts.to(device)           # [B, 3, 3, 2]
        B        = images.shape[0]

        flat_imgs = images.reshape(B * BOARD_CELLS, *images.shape[-3:])
        embs      = encoder(flat_imgs)                                  # [B*9, D]

        # digit_probs returns (shape_probs, color_probs) each [B*9, 3]
        shape_probs, color_probs = digit_probs(embs, prototypes)

        pred_shape = shape_probs.argmax(1)   # [B*9]
        pred_color = color_probs.argmax(1)   # [B*9]

        true_shape = concepts[:, :, :, 0].reshape(B * BOARD_CELLS)
        true_color = concepts[:, :, :, 1].reshape(B * BOARD_CELLS)

        correct_concept += (pred_shape == true_shape).sum().item()
        correct_concept += (pred_color == true_color).sum().item()
        total_concept   += 2 * B * BOARD_CELLS

        # Rebuild predicted [B, 3, 3, 2] concept tensor for kand_label
        pred_concepts = torch.stack([
            pred_shape.view(B, N, OBJECTS),
            pred_color.view(B, N, OBJECTS),
        ], dim=-1)                                                       # [B, 3, 3, 2]

        pred_K = sampler.kand_label(pred_concepts)                       # [B] bool
        true_K = labels.bool().to(device)                                # [B] bool

        correct_scene += (pred_K == true_K).sum().item()
        total_scene += B

        tp_k += ( pred_K &  true_K).sum().item()
        fp_k += ( pred_K & ~true_K).sum().item()
        fn_k += (~pred_K &  true_K).sum().item()
        tn_k += (~pred_K & ~true_K).sum().item()

    precision = tp_k / max(tp_k + fp_k, 1)
    recall    = tp_k / max(tp_k + fn_k, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "concept_acc": 100.0 * correct_concept / max(total_concept, 1),
        "label_acc":   100.0  * correct_scene / max(total_scene, 1), #100.0 * (tp_k + tn_k)   / max(tp_k + fp_k + fn_k + tn_k, 1),
        "label_f1":    100.0 * f1,
    }

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from src.databuilder.Kandlogic.databuilder import get_kandlogic_dataloader
    set_seed(0)
    train_loader, test_loader, proto_images, proto_labels = get_kandlogic_dataloader(b_size=64)
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)

    encoder   = PrimitivesProtoNet().to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    sampler = Sampler()

    K               = 20
    DATASET_SIZE    = len(train_loader.dataset) #type:ignore
    T = T0 = 1.0; t = 1
    sampling_epoch  = 1
    sl_weight       = 0.8

    # Candidate cache lives on device — long dtype, shape [D, K, 3, 3, 2]
    candidate_cache = torch.zeros(
        DATASET_SIZE, K, N, OBJECTS, 2,
        dtype=torch.long, device=device,
    )

    aug_images, aug_labels = augment_support_set(proto_images, proto_labels, device)

    NUM_EPOCHS = 20
    for epoch in range(NUM_EPOCHS):
        metrics = softpnet_train_one_epoch(
            encoder      = encoder,
            optimizer    = optimizer,
            aug_images   = aug_images,
            aug_labels   = aug_labels,
            unsup_loader = train_loader,
            proto_images = proto_images,
            proto_labels = proto_labels,
            device       = device,
            sampler      = sampler,
            candidate_cache = candidate_cache,
            K            = K,
            sampling_epoch = sampling_epoch,
            T            = T,
            episodes     = 50,
            k_support    = 1,
            sl_weight    = sl_weight,
            epoch        = epoch,
            n_walk_steps = 1,
        )
        scheduler.step()

        eval_metrics = evaluate(encoder, sampler, test_loader, proto_images, proto_labels, device)

        print(
            f"Epoch {epoch:3d} | "
            f"ep_loss={metrics['ep_loss']:.4f}  ep_acc={metrics['ep_acc']:.3f} | "
            f"kl={metrics['nesy_kl']:.4f}  proto={metrics['nesy_proto']:.4f} | "
            f"concept={eval_metrics['concept_acc']:.1f}%  "
            f"label={eval_metrics['label_acc']:.1f}%  "
            f"f1={eval_metrics['label_f1']:.1f}%"
        )

        if epoch >= sampling_epoch:
            T0 = max(0.01, T0 * 0.95); T = T0; t += 1