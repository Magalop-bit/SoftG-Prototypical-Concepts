"""
Visual-Sudoku (4x4) main experiment runner.

Trains all four methods -- SoftG, SoftG-K, PNet+SL and Soft-PNet -- over a fixed
list of seeds and writes the same four artifacts the other tasks produce, into
``results/Sudoku4x4/``:

    all_results.json     per-method {summary{mean,std,values}, raw{values}}
    summary_table.csv    machine-readable mean/std table
    summary_table.txt    pretty grid table (tabulate, UTF-8)
    results_figure.png   per-metric bar chart with per-seed scatter

The metric names written to disk match the existing result files exactly:

    digit_acc / digit_f1   per-cell digit accuracy / macro-F1 over the 4 classes
    board_acc / board_f1   board-validity (SAT/UNSAT) accuracy / binary F1
    train_time_s           wall-clock training time per seed

SoftG / SoftG-K are LeNet baselines trained in two stages (forward MCMC sampling
+ backward fine-tuning), exactly as in their own modules. PNet+SL and Soft-PNet
use the prototypical encoder and train on the only-SAT board loader. A single
pair of evaluators (one per backbone) computes all four metrics so the numbers
are strictly comparable across methods.

Run from the repository root:

    python -m src.test.Sudoku4x4
    python -m src.test.Sudoku4x4 --seeds 0 1 2 --n_train 300
"""

import os
import json
import time
import random
import argparse
from statistics import mean, pstdev

import numpy as np
import torch
from tabulate import tabulate

from src.backbones.Sudoku4x4.lenet import LeNet
from src.backbones.Sudoku4x4.protoencoder import (
    ProtoEncoder, compute_prototypes, digit_probs, augment_support_set,
)
from src.samplers.sudoku4x4_sampler import Sampler
from src.databuilder.Sudoku4x4.databuilder import get_mnist_sudoku4x4_dataloader

from src.models.Sudoku4x4.softg import (
    train_softg_baseline, train_softg_backward,
)
from src.models.Sudoku4x4.softgk import (
    train_softgk_baseline, train_softgk_backward,
)
from src.models.Sudoku4x4.pnet import train_one_epoch as pnet_train_one_epoch
from src.models.Sudoku4x4.softpnet import train_one_epoch as softpnet_train_one_epoch


# -- constants ---------------------------------------------------------------
NUM_DIGITS  = 4           # digit classes 1..4
BOARD_CELLS = 16          # 4 x 4
RESULTS_DIR = os.path.join("results", "Sudoku4x4")

# Methods in display order (names exactly as written to disk).
METHODS = ["SoftG", "SoftGk", "Soft-PNet", "PNet+SL"]

METRIC_KEYS   = ["digit_acc", "digit_f1", "board_acc", "board_f1", "train_time_s"]
METRIC_LABELS = {
    "digit_acc":    "Digit Acc (%)",
    "digit_f1":     "Digit F1 (%)",
    "board_acc":    "Board Acc (%)",
    "board_f1":     "Board F1 (%)",
    "train_time_s": "Train Time (s)",
}

# Fixed cell widths reproducing the committed result-file layout. tabulate adds
# 2 spaces of its own padding, so the printed segment width = content + 2; the
# committed Sudoku file uses segment widths [16, 32, 32, 32, 32, 18].
PAD_METHOD = 12
PAD_TIME   = 14
PAD_METRIC = 28


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


def _binary_f1(tp, fp, fn):
    precision = tp / max(tp + fp, 1e-8)
    recall    = tp / max(tp + fn, 1e-8)
    return 2 * precision * recall / max(precision + recall, 1e-8)


# -- evaluators (one per backbone, identical metric dicts) -------------------
@torch.no_grad()
def evaluate_proto(encoder, test_loader, proto_images, proto_labels, device, sampler):
    """Evaluator for PNet+SL / Soft-PNet (prototypical encoder)."""
    encoder.eval()
    proto_images = proto_images.to(device)
    proto_labels = proto_labels.to(device)
    prototypes   = compute_prototypes(proto_images, proto_labels, encoder)

    def predict_board_digits(imgs):
        B = imgs.size(0)
        flat = imgs.reshape(B * BOARD_CELLS, 1, 28, 28)
        probs = digit_probs(encoder(flat), prototypes)          # (B*16, 4)
        return probs.argmax(1).view(B, BOARD_CELLS) + 1         # values 1..4

    return _eval_from_board_preds(test_loader, device, sampler, predict_board_digits)


@torch.no_grad()
def evaluate_lenet(model, test_loader, device, sampler):
    """Evaluator for SoftG / SoftG-K (LeNet 4-way logits)."""
    model.eval()

    def predict_board_digits(imgs):
        B = imgs.size(0)
        flat = imgs.reshape(B * BOARD_CELLS, 1, 28, 28)
        return model(flat).argmax(1).view(B, BOARD_CELLS) + 1   # values 1..4

    return _eval_from_board_preds(test_loader, device, sampler, predict_board_digits)


def _eval_from_board_preds(test_loader, device, sampler, predict_board_digits):
    """Shared metric accumulation given a per-board digit predictor.

    Loader yields (imgs[B,4,4,1,28,28], board_label[B], digit_board[B,4,4], idx).
    """
    d_tp = torch.zeros(NUM_DIGITS); d_fp = torch.zeros(NUM_DIGITS); d_fn = torch.zeros(NUM_DIGITS)
    b_tp = b_fp = b_fn = 0

    correct_digit = total_digits = 0
    correct_board = total_boards = 0

    for (imgs, y_label, true_digits, _idx) in test_loader:
        imgs        = imgs.to(device)
        y_label     = y_label.to(device).long().view(-1)            # (B,)
        true_digits = true_digits.to(device).view(-1, BOARD_CELLS)  # (B, 16) in 1..4
        B           = imgs.size(0)

        pred_digits = predict_board_digits(imgs)                    # (B, 16) in 1..4

        correct_digit += (pred_digits.reshape(-1) == true_digits.reshape(-1)).sum().item()
        total_digits  += B * BOARD_CELLS

        pf = pred_digits.reshape(-1)
        tf = true_digits.reshape(-1)
        for d in range(1, NUM_DIGITS + 1):
            i = d - 1
            d_tp[i] += ((pf == d) & (tf == d)).sum().item()
            d_fp[i] += ((pf == d) & (tf != d)).sum().item()
            d_fn[i] += ((pf != d) & (tf == d)).sum().item()

        # Board validity via the exact SAT-cache membership check.
        pred_board_y = sampler.tensor_check(pred_digits.view(B, 4, 4)).long().to(device)
        correct_board += (pred_board_y == y_label).sum().item()
        total_boards  += B
        b_tp += ((pred_board_y == 1) & (y_label == 1)).sum().item()
        b_fp += ((pred_board_y == 1) & (y_label == 0)).sum().item()
        b_fn += ((pred_board_y == 0) & (y_label == 1)).sum().item()

    return {
        "digit_acc": 100.0 * correct_digit / max(total_digits, 1),
        "digit_f1":  100.0 * _macro_f1(d_tp, d_fp, d_fn),
        "board_acc": 100.0 * correct_board / max(total_boards, 1),
        "board_f1":  100.0 * _binary_f1(b_tp, b_fp, b_fn),
    }


# -- loaders -----------------------------------------------------------------
def _make_loaders(args, device):
    train_loader, test_loader, osat_train_loader, _osat_test, anchor = \
        get_mnist_sudoku4x4_dataloader(device, n_train=args.n_train,
                                       n_test=args.n_test, batch_size=args.b_size)
    proto_images = anchor.to(device)
    proto_labels = (torch.arange(len(proto_images)) + 1).to(device)   # {1,2,3,4}
    return train_loader, test_loader, osat_train_loader, proto_images, proto_labels


# -- per-seed training drivers -----------------------------------------------
def run_softg_seed(loaders, device, *, args):
    train_loader, test_loader, _osat, _pi, _pl = loaders
    sampler   = Sampler()
    model     = LeNet(num_classes=NUM_DIGITS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    candidate_cache = torch.zeros((len(train_loader.dataset), 4, 4),
                                  dtype=torch.long, device=device)
    T = T0 = 1.0
    t = 1
    best_state, best_board = None, -1.0

    t0 = time.perf_counter()
    for epoch in range(args.softg_stage1_epochs):
        train_softg_baseline(epoch, model, device, train_loader, optimizer,
                             candidate_cache, T, K=args.K_softg,
                             sampling_epoch=args.sampling_epoch_softg)
        m = evaluate_lenet(model, test_loader, device, sampler)
        if m["board_acc"] > best_board:
            best_board = m["board_acc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch >= args.sampling_epoch_softg:
            T0 = T0 * 0.95; t += 1; T = max(0.01, T0)

    if best_state is not None:
        model.load_state_dict(best_state)
    optimizer_bs = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(args.softg_stage2_epochs):
        train_softg_backward(model, device, train_loader, optimizer_bs, K=args.K_softg)
    train_time = time.perf_counter() - t0

    metrics = evaluate_lenet(model, test_loader, device, sampler)
    metrics["train_time_s"] = train_time
    return metrics


def run_softgk_seed(loaders, device, *, args):
    train_loader, test_loader, _osat, _pi, _pl = loaders
    sampler   = Sampler()
    model     = LeNet(num_classes=NUM_DIGITS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    candidate_cache = torch.zeros((len(train_loader.dataset), args.K_softgk, 4, 4),
                                  dtype=torch.long, device=device)
    T = T0 = 1.0
    t = 1
    best_state, best_board = None, -1.0

    t0 = time.perf_counter()
    for epoch in range(args.softg_stage1_epochs):
        train_softgk_baseline(epoch, model, device, train_loader, optimizer,
                              candidate_cache, T, K=args.K_softgk,
                              k_loss=args.k_loss, sampling_epoch=args.sampling_epoch_softgk)
        m = evaluate_lenet(model, test_loader, device, sampler)
        if m["board_acc"] > best_board:
            best_board = m["board_acc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch >= args.sampling_epoch_softgk:
            T0 = T0 * 0.95; t += 1; T = max(0.01, T0)

    if best_state is not None:
        model.load_state_dict(best_state)
    optimizer_bs = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(args.softg_stage2_epochs):
        train_softgk_backward(model, device, train_loader, optimizer_bs,
                              K=args.K_softgk, k_loss=args.k_loss)
    train_time = time.perf_counter() - t0

    metrics = evaluate_lenet(model, test_loader, device, sampler)
    metrics["train_time_s"] = train_time
    return metrics


def run_pnet_seed(loaders, device, *, args):
    train_loader, test_loader, osat_train_loader, proto_images, proto_labels = loaders
    sampler   = Sampler()
    encoder   = ProtoEncoder(x_dim=1, hid_dim=64, z_dim=64).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    aug_images, aug_labels = augment_support_set(
        proto_images, proto_labels, device, num_augmentations=9
    )

    t0 = time.perf_counter()
    for _ in range(args.proto_epochs):
        pnet_train_one_epoch(
            encoder=encoder, optimizer=optimizer,
            aug_images=aug_images, aug_labels=aug_labels,
            unsup_loader=osat_train_loader,
            proto_images=proto_images, proto_labels=proto_labels,
            device=device, episodes=args.episodes, k_support=1,
            sl_weight=args.sl_weight_pnet,
        )
        scheduler.step()
    train_time = time.perf_counter() - t0

    metrics = evaluate_proto(encoder, test_loader, proto_images, proto_labels, device, sampler)
    metrics["train_time_s"] = train_time
    return metrics


def run_softpnet_seed(loaders, device, *, args):
    train_loader, test_loader, osat_train_loader, proto_images, proto_labels = loaders
    sampler   = Sampler()
    encoder   = ProtoEncoder(x_dim=1, hid_dim=64, z_dim=64).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    aug_images, aug_labels = augment_support_set(
        proto_images, proto_labels, device, num_augmentations=9
    )
    # Cache is indexed by training-sample index; sized to the only-SAT loader.
    candidate_cache = torch.zeros((len(osat_train_loader.dataset), args.K_softpnet, 4, 4),
                                  dtype=torch.long, device=device)
    T = T0 = 1.0
    t = 1

    t0 = time.perf_counter()
    for epoch in range(args.proto_epochs):
        softpnet_train_one_epoch(
            encoder=encoder, optimizer=optimizer,
            aug_images=aug_images, aug_labels=aug_labels,
            unsup_loader=osat_train_loader,
            proto_images=proto_images, proto_labels=proto_labels,
            device=device, candidate_cache=candidate_cache, sampler=sampler,
            K=args.K_softpnet, sampling_epoch=args.sampling_epoch_softpnet, T=T,
            episodes=args.episodes, k_support=1,
            sl_weight=args.sl_weight_softpnet, epoch=epoch,
        )
        scheduler.step()
        if epoch >= args.sampling_epoch_softpnet:
            T0 = T0 * 0.95; t += 1; T = max(0.01, T0)
    train_time = time.perf_counter() - t0

    metrics = evaluate_proto(encoder, test_loader, proto_images, proto_labels, device, sampler)
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
    # Bare "\n" line endings, no trailing newline, to match the committed file.
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
    # Match committed file: UTF-8 encoded, no trailing newline.
    with open(os.path.join(out_dir, "summary_table.txt"), "w", encoding="utf-8") as f:
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

    fig.suptitle("Visual Sudoku (4x4)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "results_figure.png"), dpi=200)
    plt.close(fig)


# -- entry point -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Sudoku4x4: SoftG / SoftGk / PNet+SL / Soft-PNet runner")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--b_size", type=int, default=64)
    # prototypical methods
    p.add_argument("--proto_epochs", type=int, default=20)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--sl_weight_pnet", type=float, default=10.0)
    p.add_argument("--sl_weight_softpnet", type=float, default=15.0)
    p.add_argument("--K_softpnet", type=int, default=100)
    p.add_argument("--sampling_epoch_softpnet", type=int, default=2)
    # LeNet baselines (two-stage)
    p.add_argument("--softg_stage1_epochs", type=int, default=100)
    p.add_argument("--softg_stage2_epochs", type=int, default=40)
    p.add_argument("--K_softg", type=int, default=100)
    p.add_argument("--K_softgk", type=int, default=100)
    p.add_argument("--k_loss", type=int, default=1)
    p.add_argument("--sampling_epoch_softg", type=int, default=10)
    p.add_argument("--sampling_epoch_softgk", type=int, default=5)
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
                  f"digit_f1={metrics['digit_f1']:.2f} board_acc={metrics['board_acc']:.2f} "
                  f"board_f1={metrics['board_f1']:.2f} time={metrics['train_time_s']:.1f}s")

    ran = [m for m in METHODS if per_seed[m]]
    results = aggregate(per_seed, ran)
    write_json(results, args.out_dir)
    write_csv(results, args.out_dir, ran)
    write_txt(results, args.out_dir, ran)
    write_figure(results, args.out_dir, ran)

    print(f"\nWrote artifacts to {args.out_dir}/:")
    print("  all_results.json  summary_table.csv  summary_table.txt  results_figure.png\n")
    with open(os.path.join(args.out_dir, "summary_table.txt"), encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()