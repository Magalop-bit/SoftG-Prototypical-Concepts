"""
Kand-Logic main experiment runner.

Trains Soft-PNet and the PNet+SL baseline over a fixed list of seeds and writes,
for the Kand-Logic task, the same four artifacts the other tasks produce into
``results/Kandlogic/``:

    all_results.json     per-method {summary{mean,std,values}, raw{values}}
    summary_table.csv    machine-readable mean/std table
    summary_table.txt    pretty grid table
    results_figure.png   per-metric bar chart with per-seed scatter

Only Soft-PNet and PNet+SL are run here (the SoftG / SoftG-K baselines are left
to their own modules). The metric names written to disk match the existing
result files exactly:

    primitive_acc / primitive_f1   per-cell joint-primitive (shape*3+color, 9 classes)
    scene_acc     / scene_f1       binary Kand scene label
    train_time_s                   wall-clock training time per seed

Run from the repository root:

    python -m src.test.Kandlogic
    python -m src.test.Kandlogic --epochs 20 --K 20 --seeds 0 128 256
"""

import os
import csv
import json
import time
import random
import argparse
from statistics import mean, pstdev

import numpy as np
import torch
import torch.nn.functional as F

from src.backbones.Kandlogic.protoencoder import (
    PrimitivesProtoNet, compute_prototypes, digit_probs,
)
from src.samplers.kandlogic_sampler import Sampler
from src.databuilder.Kandlogic.databuilder import get_kandlogic_dataloader
from src.models.Kandlogic.helpers.kandlogic import (
    augment_support_set, kand_label,
)
from src.models.Kandlogic.softpnet import softpnet_train_one_epoch
from src.models.Kandlogic.pnet import pnet_train_one_epoch


# -- Kand constants ----------------------------------------------------------
N              = 3
OBJECTS        = 3
BOARD_CELLS    = N * OBJECTS      # 9
NUM_PRIMITIVES = 9               # shape * 3 + color

RESULTS_DIR = os.path.join("results", "Kandlogic")

# Methods produced by this runner, in display order.
METHODS = ["Soft-PNet", "PNet+SL"]

# Metric columns written to disk, in display order.
METRIC_KEYS   = ["primitive_acc", "primitive_f1", "scene_acc", "scene_f1", "train_time_s"]
METRIC_LABELS = {
    "primitive_acc": "Prim. Acc (%)",
    "primitive_f1":  "Prim. F1 (%)",
    "scene_acc":     "Scene Acc (%)",
    "scene_f1":      "Scene F1 (%)",
    "train_time_s":  "Train Time (s)",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# -- Unified evaluation ------------------------------------------------------
# A single evaluator shared by both methods so the reported numbers are
# strictly comparable. It returns the four metrics in result-file naming:
# primitive accuracy/macro-F1 over the 9 joint classes, and scene accuracy/F1.
@torch.no_grad()
def evaluate_full(encoder, test_loader, proto_images, proto_labels, device):
    encoder.eval()
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)

    flat_proto_labels = proto_labels[:, 0] * 3 + proto_labels[:, 1]
    prototypes        = compute_prototypes(proto_images, flat_proto_labels, encoder)

    # Primitive (per-cell joint class) accumulators.
    prim_correct = 0
    prim_total   = 0
    tp = torch.zeros(NUM_PRIMITIVES, device=device)
    fp = torch.zeros(NUM_PRIMITIVES, device=device)
    fn = torch.zeros(NUM_PRIMITIVES, device=device)

    # Scene-label accumulators.
    scene_correct = 0
    scene_total   = 0
    s_tp = s_fp = s_fn = s_tn = 0

    for images, labels, concepts, _ in test_loader:
        images   = images.to(device)
        concepts = concepts.to(device)                       # [B, 3, 3, 2]
        B        = images.shape[0]
        C, H, W  = images.shape[-3:]

        flat_imgs = images.reshape(B * BOARD_CELLS, C, H, W)
        embs      = encoder(flat_imgs)
        shape_probs, color_probs = digit_probs(embs, prototypes)

        pred_shape = shape_probs.argmax(1)                   # [B*9]
        pred_color = color_probs.argmax(1)                   # [B*9]
        pred_prim  = pred_shape * 3 + pred_color             # [B*9] joint class

        true_shape = concepts[:, :, :, 0].reshape(B * BOARD_CELLS)
        true_color = concepts[:, :, :, 1].reshape(B * BOARD_CELLS)
        true_prim  = true_shape * 3 + true_color             # [B*9]

        prim_correct += (pred_prim == true_prim).sum().item()
        prim_total   += B * BOARD_CELLS

        for c in range(NUM_PRIMITIVES):
            pred_c = (pred_prim == c)
            true_c = (true_prim == c)
            tp[c] += (pred_c & true_c).sum()
            fp[c] += (pred_c & ~true_c).sum()
            fn[c] += (~pred_c & true_c).sum()

        # Scene-level Kand label from the predicted concept grid.
        pred_concepts = torch.stack(
            [pred_shape.view(B, N, OBJECTS), pred_color.view(B, N, OBJECTS)], dim=-1
        )
        pred_K = kand_label(pred_concepts).bool().to(device)
        true_K = labels.bool().to(device)

        scene_correct += (pred_K == true_K).sum().item()
        scene_total   += B
        s_tp += ( pred_K &  true_K).sum().item()
        s_fp += ( pred_K & ~true_K).sum().item()
        s_fn += (~pred_K &  true_K).sum().item()
        s_tn += (~pred_K & ~true_K).sum().item()

    # Primitive macro-F1 over the 9 classes.
    per_class_f1 = (2 * tp) / torch.clamp(2 * tp + fp + fn, min=1.0)
    primitive_f1 = 100.0 * per_class_f1.mean().item()

    # Scene binary F1.
    precision = s_tp / max(s_tp + s_fp, 1)
    recall    = s_tp / max(s_tp + s_fn, 1)
    scene_f1  = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "primitive_acc": 100.0 * prim_correct / max(prim_total, 1),
        "primitive_f1":  primitive_f1,
        "scene_acc":     100.0 * scene_correct / max(scene_total, 1),
        "scene_f1":      100.0 * scene_f1,
    }


# -- Single-seed training drivers --------------------------------------------
def run_softpnet_seed(loaders, device, *, epochs, K, sampling_epoch, sl_weight,
                      episodes, k_support, lr, weight_decay, verbose):
    train_loader, test_loader, proto_images, proto_labels = loaders

    encoder   = PrimitivesProtoNet().to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    sampler = Sampler()
    dataset_size = len(train_loader.dataset)  # type: ignore[arg-type]
    candidate_cache = torch.zeros(
        dataset_size, K, N, OBJECTS, 2, dtype=torch.long, device=device,
    )

    aug_images, aug_labels = augment_support_set(proto_images, proto_labels, device)

    T = T0 = 1.0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        softpnet_train_one_epoch(
            encoder=encoder, optimizer=optimizer,
            aug_images=aug_images, aug_labels=aug_labels,
            unsup_loader=train_loader,
            proto_images=proto_images, proto_labels=proto_labels,
            device=device, sampler=sampler, candidate_cache=candidate_cache,
            K=K, sampling_epoch=sampling_epoch, T=T,
            episodes=episodes, k_support=k_support, sl_weight=sl_weight,
            epoch=epoch, n_walk_steps=1,
        )
        scheduler.step()
        if epoch >= sampling_epoch:
            T0 = max(0.01, T0 * 0.95)
            T = T0
        if verbose:
            m = evaluate_full(encoder, test_loader, proto_images, proto_labels, device)
            print(f"    [Soft-PNet] epoch {epoch:2d} | "
                  f"prim={m['primitive_acc']:.1f}% f1={m['primitive_f1']:.1f}% "
                  f"scene={m['scene_acc']:.1f}%")
    train_time = time.perf_counter() - t0

    metrics = evaluate_full(encoder, test_loader, proto_images, proto_labels, device)
    metrics["train_time_s"] = train_time
    return metrics


def run_pnet_seed(loaders, device, *, epochs, sl_weight, episodes, k_support,
                  lr, weight_decay, verbose):
    train_loader, test_loader, proto_images, proto_labels = loaders

    encoder   = PrimitivesProtoNet().to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    aug_images, aug_labels = augment_support_set(
        proto_images, proto_labels, device, num_augmentations=3
    )

    t0 = time.perf_counter()
    for epoch in range(epochs):
        pnet_train_one_epoch(
            encoder=encoder, optimizer=optimizer,
            aug_images=aug_images, aug_labels=aug_labels,
            unsup_loader=train_loader,
            proto_images=proto_images, proto_labels=proto_labels,
            device=device, episodes=episodes, k_support=k_support,
            sl_weight=sl_weight, epoch=epoch,
        )
        scheduler.step()
        if verbose:
            m = evaluate_full(encoder, test_loader, proto_images, proto_labels, device)
            print(f"    [PNet+SL]  epoch {epoch:2d} | "
                  f"prim={m['primitive_acc']:.1f}% f1={m['primitive_f1']:.1f}% "
                  f"scene={m['scene_acc']:.1f}%")
    train_time = time.perf_counter() - t0

    metrics = evaluate_full(encoder, test_loader, proto_images, proto_labels, device)
    metrics["train_time_s"] = train_time
    return metrics


# -- Aggregation + artifact writers ------------------------------------------
def aggregate(per_seed):
    """per_seed[method] = list of metric dicts -> summary/raw structures."""
    summary, raw = {}, {}
    for method, runs in per_seed.items():
        summary[method] = {}
        raw[method]     = {}
        for key in METRIC_KEYS:
            vals = [r[key] for r in runs]
            summary[method][key] = {
                "mean":   mean(vals),
                "std":    pstdev(vals) if len(vals) > 1 else 0.0,
                "values": vals,
            }
            raw[method][key] = vals
    return {"summary": summary, "raw": raw}


def write_json(results, out_dir):
    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def write_csv(results, out_dir):
    header = ["method"]
    for key in METRIC_KEYS:
        header += [f"{key}_mean", f"{key}_std"]
    with open(os.path.join(out_dir, "summary_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for method in results["summary"]:
            row = [method]
            for key in METRIC_KEYS:
                s = results["summary"][method][key]
                row += [f"{s['mean']:.4f}", f"{s['std']:.4f}"]
            w.writerow(row)


def _fmt_cell(key, s):
    if key == "train_time_s":
        return f"{s['mean']:.1f}s \u00b1 {s['std']:.1f}s"
    return f"{s['mean']:.2f} \u00b1 {s['std']:.2f}"


def write_txt(results, out_dir):
    cols = ["Method"] + [METRIC_LABELS[k] for k in METRIC_KEYS]
    rows = []
    for method in results["summary"]:
        cells = [method] + [
            _fmt_cell(k, results["summary"][method][k]) for k in METRIC_KEYS
        ]
        rows.append(cells)

    widths = [len(c) for c in cols]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def sep(fill):
        return "+" + "+".join(fill * (w + 2) for w in widths) + "+"

    def line(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [sep("-"), line(cols), sep("=")]
    for r in rows:
        out.append(line(r))
        out.append(sep("-"))

    with open(os.path.join(out_dir, "summary_table.txt"), "w") as f:
        f.write("\n".join(out) + "\n")


def write_figure(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_keys = [k for k in METRIC_KEYS if k != "train_time_s"]
    methods   = list(results["summary"].keys())
    colors    = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 3)))

    fig, axes = plt.subplots(1, len(plot_keys), figsize=(5 * len(plot_keys), 5))
    if len(plot_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, plot_keys):
        means = [results["summary"][m][key]["mean"] for m in methods]
        stds  = [results["summary"][m][key]["std"]  for m in methods]
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, capsize=6, color=colors[:len(methods)],
               alpha=0.85, edgecolor="black", linewidth=0.6)
        for i, m in enumerate(methods):
            vals = results["summary"][m][key]["values"]
            jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color="black", s=14, alpha=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_title(METRIC_LABELS[key])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Kand-Logic - Soft-PNet vs PNet+SL", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "results_figure.png"), dpi=200)
    plt.close(fig)


# -- Entry point -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Kand-Logic: Soft-PNet & PNet+SL runner")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[i * 128 for i in range(10)])
    p.add_argument("--K", type=int, default=20, help="Soft-PNet cache working-set size")
    p.add_argument("--sampling_epoch", type=int, default=1)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--k_support", type=int, default=1)
    p.add_argument("--sl_weight", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--b_size", type=int, default=64)
    p.add_argument("--out_dir", type=str, default=RESULTS_DIR)
    p.add_argument("--verbose", action="store_true", help="print per-epoch eval")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | seeds={args.seeds} | epochs={args.epochs} | K={args.K}")

    per_seed = {m: [] for m in METHODS}

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        # Re-seed and rebuild loaders so both methods see identical data/order.
        set_seed(seed)
        loaders = get_kandlogic_dataloader(b_size=args.b_size)
        train_loader, test_loader, proto_images, proto_labels = loaders
        proto_images = proto_images.to(device)
        proto_labels = proto_labels.to(device)
        loaders = (train_loader, test_loader, proto_images, proto_labels)

        set_seed(seed)
        sp = run_softpnet_seed(
            loaders, device,
            epochs=args.epochs, K=args.K, sampling_epoch=args.sampling_epoch,
            sl_weight=args.sl_weight, episodes=args.episodes,
            k_support=args.k_support, lr=args.lr, weight_decay=args.weight_decay,
            verbose=args.verbose,
        )
        per_seed["Soft-PNet"].append(sp)
        print(f"  Soft-PNet: prim_acc={sp['primitive_acc']:.2f} "
              f"prim_f1={sp['primitive_f1']:.2f} scene_acc={sp['scene_acc']:.2f} "
              f"scene_f1={sp['scene_f1']:.2f} time={sp['train_time_s']:.1f}s")

        set_seed(seed)
        pn = run_pnet_seed(
            loaders, device,
            epochs=args.epochs, sl_weight=args.sl_weight, episodes=args.episodes,
            k_support=args.k_support, lr=args.lr, weight_decay=args.weight_decay,
            verbose=args.verbose,
        )
        per_seed["PNet+SL"].append(pn)
        print(f"  PNet+SL  : prim_acc={pn['primitive_acc']:.2f} "
              f"prim_f1={pn['primitive_f1']:.2f} scene_acc={pn['scene_acc']:.2f} "
              f"scene_f1={pn['scene_f1']:.2f} time={pn['train_time_s']:.1f}s")

    results = aggregate(per_seed)
    write_json(results, args.out_dir)
    write_csv(results, args.out_dir)
    write_txt(results, args.out_dir)
    write_figure(results, args.out_dir)

    print(f"\nWrote artifacts to {args.out_dir}/:")
    print("  all_results.json  summary_table.csv  summary_table.txt  results_figure.png\n")
    with open(os.path.join(args.out_dir, "summary_table.txt")) as f:
        print(f.read())


if __name__ == "__main__":
    main()