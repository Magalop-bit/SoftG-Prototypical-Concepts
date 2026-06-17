import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, TensorDataset

from src.models.Kandlogic.helpers.kandlogic_sl import kand_nesy_loss
from src.models.Kandlogic.helpers.kandlogic import kand_label, _row_predicate, augment_support_set, split_support_query

from src.backbones.Kandlogic.protoencoder import (PrimitivesProtoNet, prototypical_loss, digit_probs, compute_prototypes)

figures = {0: 'square', 1: 'circle', 2:'triangle'}
colors = {0:'red', 1:'yellow', 2:'blue'}

_PRIMITIVE_LABELS = {
    0: 'red-square', 1: 'red-circle', 2:'red-triangle',
    3: 'yellow-square', 4: 'yellow-circle', 5:'yellow-triangle',
    6: 'blue-square', 7: 'blue-circle', 8:'blue-triangle',
}

def pnet_train_one_epoch(
    encoder:         PrimitivesProtoNet,
    optimizer:       torch.optim.Optimizer,
    aug_images:      torch.Tensor,   # (N*(1+num_aug), C, H, W)  joint-encoded
    aug_labels:      torch.Tensor,   # (N*(1+num_aug),)           flat joint labels
    unsup_loader:    DataLoader,
    proto_images:    torch.Tensor,
    proto_labels:    torch.Tensor,
    device:          torch.device,
    episodes:        int   = 50,
    k_support:       int   = 1,
    sl_weight:       float = 10.0,
    epoch:           int   = 0,
) -> dict:
    encoder.train()
    aug_images  = aug_images.to(device)
    aug_labels  = aug_labels.to(device)
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)

    # -----------------------------------------------------------------------
    # PHASE 1 — episodic prototypical loss on the augmented support set
    # -----------------------------------------------------------------------
    ep_losses, ep_accs = [], []

    for ep in range(episodes):
        # draw a fresh support/query split each episode
        s_img, s_lbl, q_img, q_lbl = split_support_query(
            aug_images, aug_labels, k_support
        )

        # skip degenerate episodes (no query samples)
        if q_img.shape[0] == 0:
            continue

        # embed support + query in one forward pass so gradients flow through both
        all_emb = encoder(torch.cat([s_img, q_img], dim=0))   # (S+Q, D)
        all_lbl = torch.cat([s_lbl, q_lbl], dim=0)            # (S+Q,)

        optimizer.zero_grad()
        loss, acc = prototypical_loss(all_emb, all_lbl, n_support=k_support)
        loss.backward()
        optimizer.step()

        ep_losses.append(loss.item())
        ep_accs.append(acc.item())

    # -----------------------------------------------------------------------
    # PHASE 2 — joint prototypical + Semantic Loss on the unsupervised pairs
    # -----------------------------------------------------------------------
    # PHASE 2 — NeSy loss on the unsupervised loader
    nesy_losses, nesy_sl, nesy_proto = [], [], []

    for images, labels, _, _ in unsup_loader:
        images = images.to(device)
        # scene label: 1 = positive Kand pattern, 0 = negative
        # kand_label expects concepts on CPU for the bool ops
        y_scene = labels.float().to(device)
        #y_scene = kand_label(concepts).to(device) 

        flat_proto_labels = proto_labels[:, 0] * 3 + proto_labels[:, 1]
        prototypes = compute_prototypes(proto_images, flat_proto_labels, encoder)

        optimizer.zero_grad()

        # episodic proto loss on a fresh split (keeps proto training alive)
        s_img, s_lbl, q_img, q_lbl = split_support_query(aug_images, aug_labels, k_support)
        if q_img.shape[0] > 0:
            ep_emb = encoder(torch.cat([s_img, q_img]))
            ep_lbl = torch.cat([s_lbl, q_lbl])
            loss_proto, _ = prototypical_loss(ep_emb, ep_lbl, n_support=k_support)
        else:
            loss_proto = torch.tensor(0.0, device=device)

        loss_sl, proto_pull = kand_nesy_loss(images, y_scene, encoder, prototypes, device)

        loss = loss_proto + sl_weight * loss_sl #+ 0.01 * proto_pull
        loss.backward()
        optimizer.step()

        nesy_losses.append(loss.item())
        nesy_sl.append(loss_sl.item())
        nesy_proto.append(loss_proto.item())

    # TODO: iterate unsup_loader, compute prototypes from proto_images/proto_labels,
    #       build digit_probs, assemble the NeSy constraint, combine with
    #       sl_weight * loss_sl + loss_proto, backward + step.

    # -----------------------------------------------------------------------
    # PHASE 3 — (future) curriculum / scheduler hook
    # -----------------------------------------------------------------------
    # TODO: e.g. anneal sl_weight, step LR scheduler, log to wandb, etc.

    return {
        "ep_loss":    sum(ep_losses)    / max(len(ep_losses),   1),
        "ep_acc":     sum(ep_accs)      / max(len(ep_accs),     1),
        "nesy_loss":  sum(nesy_losses)  / max(len(nesy_losses), 1),
        "nesy_sl":    sum(nesy_sl)      / max(len(nesy_sl),     1),
        "nesy_proto": sum(nesy_proto)   / max(len(nesy_proto),  1),
    }

@torch.no_grad()
def evaluate(
    encoder:      PrimitivesProtoNet,
    test_loader:  DataLoader,
    proto_images: torch.Tensor,
    proto_labels: torch.Tensor,
    device:       torch.device,
) -> dict:
    """
    Metrics:
      concept_acc  – per-primitive classification accuracy (shape + color)
      label_acc    – scene-level Kand label accuracy (predicted K vs true K)
      label_f1     – macro F1 on the binary scene label
    """
    encoder.eval()
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)
    prototypes   = compute_prototypes(proto_images,  proto_labels[:, 0] * 3 + proto_labels[:, 1], encoder)

    correct_concept = 0
    correct_label   = 0
    total_concept   = 0
    total_label     = 0

    tp_k = fp_k = fn_k = tn_k = 0

    for images, labels, concepts, idxs in test_loader:
        # images   : (B, N, N, C, H, W)
        # concepts : (B, N, N, 2)   [shape, color]
        images   = images.to(device)
        concepts = concepts.to(device)
        B, N, _, C, H, W = images.shape

        # ── concept predictions ───────────────────────────────────────────
        flat_imgs = images.view(B * N * N, C, H, W)
        embs                    = encoder(flat_imgs)
        shape_probs, color_probs = digit_probs(embs, prototypes)

        pred_shape = shape_probs.argmax(1)   # (B*9,)
        pred_color = color_probs.argmax(1)   # (B*9,)

        true_shape = concepts[:, :, :, 0].view(B * N * N)
        true_color = concepts[:, :, :, 1].view(B * N * N)

        correct_concept += (pred_shape == true_shape).sum().item()
        correct_concept += (pred_color == true_color).sum().item()
        total_concept   += 2 * B * N * N

        # ── scene-level Kand label ────────────────────────────────────────
        # rebuild predicted concepts tensor (B, N, N, 2) for kand_label
        pred_concepts = torch.stack([
            pred_shape.view(B, N, N),
            pred_color.view(B, N, N),
        ], dim=-1)                                            # (B, N, N, 2)

        pred_K = kand_label(pred_concepts).bool()           # (B,) bool
        true_K = labels.to(device).bool()                   # (B,) bool

        correct_label += (pred_K == true_K).sum().item()
        total_label += B

        tp_k += ( pred_K &  true_K).sum().item()
        fp_k += ( pred_K & ~true_K).sum().item()
        fn_k += (~pred_K &  true_K).sum().item()
        tn_k += (~pred_K & ~true_K).sum().item()

    # ── aggregate ─────────────────────────────────────────────────────────
    precision = tp_k / max(tp_k + fp_k, 1)
    recall    = tp_k / max(tp_k + fn_k, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "concept_acc": 100.0 * correct_concept / max(total_concept, 1),
        "label_acc":   100.0 * correct_label / max(total_label, 1),
        "label_f1":    100.0 * f1,
    }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from src.databuilder.Kandlogic.databuilder import get_kandlogic_dataloader
    train_loader, test_loader, proto_images, proto_labels = get_kandlogic_dataloader(b_size=64)
    proto_images, proto_labels = proto_images.to(device), proto_labels.to(device)

    aug_images, aug_labels = augment_support_set(proto_images, proto_labels, device, num_augmentations=3)

    s_imgs, s_lbls, q_imgs, g_lbls = split_support_query(aug_images, aug_labels, k_support=1)

    encoder = PrimitivesProtoNet().to(device)
    optimizer   = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    num_epochs = 10
    episodes = 50
    sl_weight = 10.0
    
    if True:
        for epoch in range(num_epochs):
            stats = pnet_train_one_epoch(
                encoder      = encoder,
                optimizer    = optimizer,
                aug_images   = aug_images,
                aug_labels   = aug_labels,
                unsup_loader = train_loader,
                proto_images = proto_images,
                proto_labels = proto_labels,
                device       = device,
                episodes     = episodes,
                k_support    = 1,
                sl_weight    = sl_weight,
                epoch        = epoch,
            )
            scheduler.step()

            eval_stats = evaluate(encoder, test_loader, proto_images, proto_labels, device)

            print(
                f"Epoch {epoch:02d}/{num_epochs} | "
                f"ep_loss={stats['ep_loss']:.4f} ep_acc={stats['ep_acc']:.4f} | "
                f"nesy_loss={stats['nesy_loss']:.4f} "
                f"(sl={stats['nesy_sl']:.4f} proto={stats['nesy_proto']:.4f}) | "
                f"concept_acc={eval_stats['concept_acc']:.1f}% "
                f"label_acc={eval_stats['label_acc']:.1f}% "
                f"label_f1={eval_stats['label_f1']:.1f}%"
            )