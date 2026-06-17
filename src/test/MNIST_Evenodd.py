"""
MNIST-EvenOdd main experiment runner.

Trains all four methods -- SoftG, SoftG-K, PNet+SL and Soft-PNet -- over a fixed
list of seeds and writes the same four artifacts the other tasks produce, into
``results/MNIST_Evenodd/``:

    all_results.json     per-method {summary{mean,std,values}, raw{values}}
    summary_table.csv    machine-readable mean/std table
    summary_table.txt    pretty grid table (tabulate, latin-1)
    results_figure.png   per-metric bar chart with per-seed scatter

The metric names written to disk match the existing result files exactly:

    digit_acc / digit_f1   per-digit concept accuracy / macro-F1 over 10 classes
    sum_acc   / sum_f1     pair-sum label accuracy / macro-F1 over sum classes
    train_time_s           wall-clock training time per seed

SoftG / SoftG-K are LeNet baselines trained in two stages (forward MCMC sampling
+ backward fine-tuning), exactly as in their own modules. PNet+SL and Soft-PNet
use the prototypical encoder. A single pair of evaluators (one per backbone)
computes all four metrics so the numbers are strictly comparable across methods.

Run from the repository root:

    python -m src.test.MNIST_Evenodd
    python -m src.test.MNIST_Evenodd --seeds 0 1 2 --n_train 6720
"""

import os
import json
import math
import time
import random
import argparse
from statistics import mean, pstdev

import numpy as np
import torch
from tabulate import tabulate

from src.backbones.MNIST_Evenodd.lenet import LeNet
from src.backbones.MNIST_Evenodd.protoencoder import (
    ProtoEncoder, compute_prototypes, digit_probs, augment_support_set,
)
from src.samplers.mnistevenodd_sampler import Sampler
from src.databuilder.MNIST_Evenodd.databuilder import get_mnist_evenodd_dataloader

from src.models.MNIST_Evenodd.softg import (
    softg_train_one_epoch, train_softg_backward,
)
from src.models.MNIST_Evenodd.softgk import (
    softgk_train_one_epoch, train_softgk_backward,
)
from src.models.MNIST_Evenodd.pnet import pnet_train_one_epoch
from src.models.MNIST_Evenodd.softpnet import softpnet_train_one_epoch


# -- constants ---------------------------------------------------------------
NUM_DIGITS = 10
MAX_SUM    = 18                       # largest digit-pair sum (9 + 9)
RESULTS_DIR = os.path.join("results", "MNIST_Evenodd")

# Methods in display order (names exactly as written to disk).
METHODS = ["SoftG", "SoftGk", "Soft-PNet", "PNet+SL"]

METRIC_KEYS   = ["digit_acc", "digit_f1", "sum_acc", "sum_f1", "train_time_s"]
METRIC_LABELS = {
    "digit_acc":    "Digit Acc (%)",
    "digit_f1":     "Digit F1 (%)",
    "sum_acc":      "Sum Acc (%)",
    "sum_f1":       "Sum F1 (%)",
    "train_time_s": "Train Time (s)",
}

# Fixed cell widths reproducing the committed result-file layout (tabulate adds
# 2 spaces of its own padding, so segment width = content width + 2).
PAD_METHOD = 12
PAD_TIME   = 14
PAD_METRIC = 26


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# -- metric helper -----------------------------------------------------------
def _macro_f1(tp, fp, fn):
    precision = tp / (tp + fp).clamp(min=1e-8)
    recall    = tp / (tp + fn).clamp(min=1e-8)
    f1        = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return f1.mean().item()


# -- evaluators (one per backbone, identical metric dicts) -------------------
@torch.no_grad()
def evaluate_proto(encoder, test_loader, proto_images, proto_labels, device):
    """Evaluator for PNet+SL / Soft-PNet (prototypical encoder)."""
    encoder.eval()
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)
    prototypes   = compute_prototypes(proto_images, proto_labels, encoder)
    return _eval_from_digit_preds(
        test_loader, device,
        lambda img: digit_probs(encoder(img.to(device)), prototypes).argmax(1),
    )


@torch.no_grad()
def evaluate_lenet(model, test_loader, device):
    """Evaluator for SoftG / SoftG-K (LeNet 10-way logits)."""
    model.eval()
    return _eval_from_digit_preds(
        test_loader, device,
        lambda img: model(img.to(device)).argmax(1),
    )


def _eval_from_digit_preds(test_loader, device, predict_digits):
    """Shared metric accumulation given a per-image digit predictor."""
    d_tp = torch.zeros(NUM_DIGITS); d_fp = torch.zeros(NUM_DIGITS); d_fn = torch.zeros(NUM_DIGITS)
    s_tp = torch.zeros(MAX_SUM + 1); s_fp = torch.zeros(MAX_SUM + 1); s_fn = torch.zeros(MAX_SUM + 1)

    correct_digit = correct_sum = total_digits = total_pairs = 0

    for (images, y_sum, true_d1, true_d2, _) in test_loader:
        img1, img2 = images[:, 0], images[:, 1]
        true_d1 = true_d1.to(device)
        true_d2 = true_d2.to(device)
        y_sum   = y_sum.to(device)

        pred_d1  = predict_digits(img1)
        pred_d2  = predict_digits(img2)
        pred_sum = pred_d1 + pred_d2

        correct_sum   += (pred_sum == y_sum).sum().item()
        total_pairs   += y_sum.size(0)
        correct_digit += (pred_d1 == true_d1).sum().item()
        correct_digit += (pred_d2 == true_d2).sum().item()
        total_digits  += 2 * y_sum.size(0)

        for d in range(NUM_DIGITS):
            for pred, true in ((pred_d1, true_d1), (pred_d2, true_d2)):
                d_tp[d] += ((pred == d) & (true == d)).sum().item()
                d_fp[d] += ((pred == d) & (true != d)).sum().item()
                d_fn[d] += ((pred != d) & (true == d)).sum().item()

        for s in range(MAX_SUM + 1):
            s_tp[s] += ((pred_sum == s) & (y_sum == s)).sum().item()
            s_fp[s] += ((pred_sum == s) & (y_sum != s)).sum().item()
            s_fn[s] += ((pred_sum != s) & (y_sum == s)).sum().item()

    return {
        "digit_acc": 100.0 * correct_digit / max(total_digits, 1),
        "digit_f1":  100.0 * _macro_f1(d_tp, d_fp, d_fn),
        "sum_acc":   100.0 * correct_sum / max(total_pairs, 1),
        "sum_f1":    100.0 * _macro_f1(s_tp, s_fp, s_fn),
    }


# -- per-seed training drivers -----------------------------------------------
def _make_loaders(args, device):
    train_loader, test_loader, anchor = get_mnist_evenodd_dataloader(
        n_train=args.n_train, n_test=args.n_test, b_size=args.b_size,
    )
    proto_images = anchor.to(device)
    proto_labels = torch.arange(len(proto_images)).to(device)
    return train_loader, test_loader, proto_images, proto_labels


def run_softg_seed(loaders, device, *, args):
    train_loader, test_loader, _, _ = loaders
    model     = LeNet(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler   = Sampler()

    candidate_cache = torch.zeros((len(train_loader.dataset), 2),
                                  dtype=torch.long, device=device)
    T = T0 = 1.0
    t = 1
    best_state = None
    best_sum = -1.0

    t0 = time.perf_counter()
    # Stage-I: forward sampling with MCMC + exponential annealing.
    for epoch in range(args.softg_stage1_epochs):
        softg_train_one_epoch(epoch, model, sampler, device, train_loader,
                              optimizer, candidate_cache, T, K=args.K_softg,
                              sampling_epoch=args.sampling_epoch_softg)
        m = evaluate_lenet(model, test_loader, device)
        if m["sum_acc"] > best_sum:
            best_sum = m["sum_acc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch >= args.sampling_epoch_softg:
            T0 = T0 * 0.95
            t += 1
            T = max(0.01, T0)

    # Stage-II: backward fine-tuning from the best Stage-I checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)
    optimizer_bs = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(args.softg_stage2_epochs):
        train_softg_backward(model, sampler, device, train_loader,
                             optimizer_bs, K=args.K_softg)
    train_time = time.perf_counter() - t0

    metrics = evaluate_lenet(model, test_loader, device)
    metrics["train_time_s"] = train_time
    return metrics


def run_softgk_seed(loaders, device, *, args):
    train_loader, test_loader, _, _ = loaders
    model     = LeNet(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler   = Sampler()

    candidate_cache = torch.zeros((len(train_loader.dataset), args.K_softg, 2),
                                  dtype=torch.long, device=device)
    T = T0 = 1.0
    t = 1
    best_state = None
    best_sum = -1.0

    t0 = time.perf_counter()
    for epoch in range(args.softg_stage1_epochs):
        softgk_train_one_epoch(epoch, model, sampler, device, train_loader,
                               optimizer, candidate_cache, T, K=args.K_softg,
                               sampling_epoch=args.sampling_epoch_softg)
        m = evaluate_lenet(model, test_loader, device)
        if m["sum_acc"] > best_sum:
            best_sum = m["sum_acc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch >= args.sampling_epoch_softg:
            T0 = T0 * 0.95
            t += 1
            T = max(0.01, T0)

    if best_state is not None:
        model.load_state_dict(best_state)
    optimizer_bs = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(args.softg_stage2_epochs):
        train_softgk_backward(model, sampler, device, train_loader,
                              optimizer_bs, K=args.K_softg)
    train_time = time.perf_counter() - t0

    metrics = evaluate_lenet(model, test_loader, device)
    metrics["train_time_s"] = train_time
    return metrics


def run_pnet_seed(loaders, device, *, args):
    train_loader, test_loader, proto_images, proto_labels = loaders
    encoder   = ProtoEncoder(x_dim=1, hid_dim=64, z_dim=64).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    aug_images, aug_labels = augment_support_set(
        proto_images, proto_labels, device, num_augmentations=9
    )

    t0 = time.perf_counter()
    for epoch in range(args.proto_epochs):
        pnet_train_one_epoch(
            encoder=encoder, optimizer=optimizer,
            aug_images=aug_images, aug_labels=aug_labels,
            unsup_loader=train_loader,
            proto_images=proto_images, proto_labels=proto_labels,
            device=device, episodes=args.episodes, k_support=1,
            sl_weight=args.sl_weight_pnet, epoch=epoch,
        )
        scheduler.step()
    train_time = time.perf_counter() - t0

    metrics = evaluate_proto(encoder, test_loader, proto_images, proto_labels, device)
    metrics["train_time_s"] = train_time
    return metrics


def run_softpnet_seed(loaders, device, *, args):
    train_loader, test_loader, proto_images, proto_labels = loaders
    encoder   = ProtoEncoder(x_dim=1, hid_dim=64, z_dim=64).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    sampler = Sampler()
    aug_images, aug_labels = augment_support_set(
        proto_images, proto_labels, device, num_augmentations=9
    )
    candidate_cache = torch.zeros((len(train_loader.dataset), args.K_softpnet, 2),
                                  dtype=torch.long, device=device)
    T = T0 = 1.0
    t = 1

    t0 = time.perf_counter()
    for epoch in range(args.proto_epochs):
        softpnet_train_one_epoch(
            encoder=encoder, optimizer=optimizer,
            aug_images=aug_images, aug_labels=aug_labels,
            unsup_loader=train_loader,
            proto_images=proto_images, proto_labels=proto_labels,
            device=device, candidate_cache=candidate_cache, sampler=sampler,
            K=args.K_softpnet, sampling_epoch=args.sampling_epoch_softpnet, T=T,
            episodes=args.episodes, k_support=1,
            sl_weight=args.sl_weight_softpnet, epoch=epoch,
        )
        scheduler.step()
        if epoch >= args.sampling_epoch_softpnet:
            T0 = T0 * 0.95
            t += 1
            T = max(0.01, T0)
    train_time = time.perf_counter() - t0

    metrics = evaluate_proto(encoder, test_loader, proto_images, proto_labels, device)
    metrics["train_time_s"] = train_time
    return metrics


RUNNERS = {
    "SoftG":     run_softg_seed,
    "SoftGk":    run_softgk_seed,
    "Soft-PNet": run_softpnet_seed,
    "PNet+SL":   run_pnet_seed,
}


# -- aggregation + artifact writers ------------------------------------------
def aggregate(per_seed, methods):
    summary, raw = {}, {}
    for method in methods:
        runs = per_seed[method]
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


def write_csv(results, out_dir, methods):
    # Written manually with bare "\n" line endings and no trailing newline, to
    # match the committed files (csv.writer would emit "\r\n").
    header = ["method"]
    for key in METRIC_KEYS:
        header += [f"{key}_mean", f"{key}_std"]
    lines = [",".join(header)]
    for method in methods:
        row = [method]
        for key in METRIC_KEYS:
            s = results["summary"][method][key]
            row += [f"{s['mean']:.4f}", f"{s['std']:.4f}"]
        lines.append(",".join(row))
    with open(os.path.join(out_dir, "summary_table.csv"), "w") as f:
        f.write("\n".join(lines))


def _fmt_cell(key, s):
    if key == "train_time_s":
        return f"{s['mean']:.1f} \u00b1 {s['std']:.1f}"
    return f"{s['mean']:.2f} \u00b1 {s['std']:.2f}"


def _pad(key, text):
    if key == "method":
        return text.ljust(PAD_METHOD)
    if key == "train_time_s":
        return text.ljust(PAD_TIME)
    return text.ljust(PAD_METRIC)


def write_txt(results, out_dir, methods):
    headers = [_pad("method", "Method")] + [_pad(k, METRIC_LABELS[k]) for k in METRIC_KEYS]
    rows = []
    for method in methods:
        cells = [_pad("method", method)]
        cells += [_pad(k, _fmt_cell(k, results["summary"][method][k])) for k in METRIC_KEYS]
        rows.append(cells)
    table = tabulate(rows, headers=headers, tablefmt="grid")
    # Match committed files: latin-1 encoded, no trailing newline.
    with open(os.path.join(out_dir, "summary_table.txt"), "w", encoding="latin-1") as f:
        f.write(table)


def write_figure(results, out_dir, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_keys = [k for k in METRIC_KEYS if k != "train_time_s"]
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

    fig.suptitle("MNIST-EvenOdd", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "results_figure.png"), dpi=200)
    plt.close(fig)


# -- entry point -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="MNIST-EvenOdd: SoftG / SoftGk / PNet+SL / Soft-PNet runner")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n_train", type=int, default=6720)
    p.add_argument("--n_test", type=int, default=960)
    p.add_argument("--b_size", type=int, default=64)
    # prototypical methods
    p.add_argument("--proto_epochs", type=int, default=10)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--sl_weight_pnet", type=float, default=10.0)
    p.add_argument("--sl_weight_softpnet", type=float, default=5.0)
    p.add_argument("--K_softpnet", type=int, default=20)
    p.add_argument("--sampling_epoch_softpnet", type=int, default=1)
    # LeNet baselines (two-stage)
    p.add_argument("--softg_stage1_epochs", type=int, default=10)
    p.add_argument("--softg_stage2_epochs", type=int, default=40)
    p.add_argument("--K_softg", type=int, default=10)
    p.add_argument("--sampling_epoch_softg", type=int, default=10)
    p.add_argument("--out_dir", type=str, default=RESULTS_DIR)
    p.add_argument("--methods", type=str, nargs="+", default=METHODS,
                   choices=METHODS, help="subset of methods to run")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | seeds={args.seeds} | methods={args.methods}")

    per_seed = {m: [] for m in METHODS}

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        for method in args.methods:
            # Re-seed and rebuild loaders before each method so every method
            # sees identical data and ordering for a given seed.
            set_seed(seed)
            loaders = _make_loaders(args, device)
            set_seed(seed)
            metrics = RUNNERS[method](loaders, device, args=args)
            per_seed[method].append(metrics)
            print(f"  {method:10s}: digit_acc={metrics['digit_acc']:.2f} "
                  f"digit_f1={metrics['digit_f1']:.2f} sum_acc={metrics['sum_acc']:.2f} "
                  f"sum_f1={metrics['sum_f1']:.2f} time={metrics['train_time_s']:.1f}s")

    # Only aggregate and write methods that were actually run, preserving order.
    ran = [m for m in METHODS if per_seed[m]]

    results = aggregate(per_seed, ran)
    write_json(results, args.out_dir)
    write_csv(results, args.out_dir, ran)
    write_txt(results, args.out_dir, ran)
    write_figure(results, args.out_dir, ran)

    print(f"\nWrote artifacts to {args.out_dir}/:")
    print("  all_results.json  summary_table.csv  summary_table.txt  results_figure.png\n")
    with open(os.path.join(args.out_dir, "summary_table.txt"), encoding="latin-1") as f:
        print(f.read())


if __name__ == "__main__":
    main()