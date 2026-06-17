"""
Kandinsky SAT Scenario Generator
=================================
Scene array shape: [3, 3, 2]
  - [:, :, 0]  -> shapes   (values in {0,1,2})
  - [:, :, 1]  -> colors   (values in {0,1,2})

Each scene x_i is represented as:
  shapes_i = (s0, s1, s2)   colors_i = (c0, c1, c2)

Predicates for scene x_i
------------------------
  SameS_i  : all 3 shapes equal
  DiffS_i  : all 3 shapes different (a permutation of {0,1,2})
  TwoS_i   : exactly 2 of 3 shapes are equal  (the remaining case)

  SameC_i  : all 3 colors equal
  DiffC_i  : all 3 colors different
  TwoC_i   : exactly 2 of 3 colors are equal

Background Knowledge K (the label is True iff one of these clauses holds):
  (DiffS1 ∧ DiffS2 ∧ DiffS3)  ∨
  (TwoS1  ∧ TwoS2  ∧ TwoS3 )  ∨
  (SameS1 ∧ SameS2 ∧ SameS3)  ∨
  (DiffC1 ∧ DiffC2 ∧ DiffC3)  ∨
  (TwoC1  ∧ TwoC2  ∧ TwoC3 )  ∨
  (SameC1 ∧ SameC2 ∧ SameC3)

Usage
-----
  python kandinsky_sat.py                              # count SAT scenarios
  python kandinsky_sat.py --enumerate                  # list all SAT scenarios
  python kandinsky_sat.py --sample 10                  # generate 10 random SAT scenarios
  python kandinsky_sat.py --sample 10 --seed 42        # reproducible
  python kandinsky_sat.py --sample 1000 --output data/sat_1000.pt   # save to .pt
"""

import argparse
import itertools
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Scene = Tuple[Tuple[int, int, int], Tuple[int, int, int]]
# Scene = (shapes, colors)  each is a 3-tuple with values in {0,1,2}

Scenario = Tuple[Scene, Scene, Scene]


# ---------------------------------------------------------------------------
# Scene predicates
# ---------------------------------------------------------------------------

def same(triple: Tuple[int, int, int]) -> bool:
    return triple[0] == triple[1] == triple[2]

def diff(triple: Tuple[int, int, int]) -> bool:
    return len(set(triple)) == 3

def two(triple: Tuple[int, int, int]) -> bool:
    """Exactly two of the three values are the same."""
    counts = [triple.count(v) for v in set(triple)]
    return sorted(counts) == [1, 2]


def scene_predicates(scene: Scene):
    shapes, colors = scene
    return {
        "SameS": same(shapes), "DiffS": diff(shapes), "TwoS": two(shapes),
        "SameC": same(colors), "DiffC": diff(colors), "TwoC": two(colors),
    }


# ---------------------------------------------------------------------------
# K evaluation
# ---------------------------------------------------------------------------

CLAUSES = [
    ("DiffS", "DiffS", "DiffS"),
    ("TwoS",  "TwoS",  "TwoS"),
    ("SameS", "SameS", "SameS"),
    ("DiffC", "DiffC", "DiffC"),
    ("TwoC",  "TwoC",  "TwoC"),
    ("SameC", "SameC", "SameC"),
]
CLAUSE_NAMES = [
    "DiffS1∧DiffS2∧DiffS3",
    "TwoS1∧TwoS2∧TwoS3",
    "SameS1∧SameS2∧SameS3",
    "DiffC1∧DiffC2∧DiffC3",
    "TwoC1∧TwoC2∧TwoC3",
    "SameC1∧SameC2∧SameC3",
]


def satisfies_K(s1: Scene, s2: Scene, s3: Scene) -> bool:
    p1, p2, p3 = scene_predicates(s1), scene_predicates(s2), scene_predicates(s3)
    for c1, c2, c3 in CLAUSES:
        if p1[c1] and p2[c2] and p3[c3]:
            return True
    return False


def which_clauses(s1: Scene, s2: Scene, s3: Scene) -> List[str]:
    """Return the names of all satisfied clauses (can be >1 simultaneously)."""
    p1, p2, p3 = scene_predicates(s1), scene_predicates(s2), scene_predicates(s3)
    satisfied = []
    for (c1, c2, c3), name in zip(CLAUSES, CLAUSE_NAMES):
        if p1[c1] and p2[c2] and p3[c3]:
            satisfied.append(name)
    return satisfied


# ---------------------------------------------------------------------------
# Universe of scenes (3^3 shapes × 3^3 colors = 729)
# ---------------------------------------------------------------------------

ALL_TRIPLES = list(itertools.product(range(3), repeat=3))
ALL_SCENES: List[Scene] = [
    (shapes, colors)
    for shapes in ALL_TRIPLES
    for colors in ALL_TRIPLES
]    #type: ignore # 729 scenes


# ---------------------------------------------------------------------------
# Enumerate ALL SAT scenarios (brute-force over 729^3 ≈ 387 M)
# Use numpy for speed instead of pure Python triple loop.
# ---------------------------------------------------------------------------

def enumerate_all_sat() -> List[Scenario]:
    """
    Returns all (s1, s2, s3) triples from ALL_SCENES that satisfy K.
    Memory note: result size can be large — use count_sat() if you only need
    the count.
    """
    sat = []
    for s1 in ALL_SCENES:
        for s2 in ALL_SCENES:
            for s3 in ALL_SCENES:
                if satisfies_K(s1, s2, s3):
                    sat.append((s1, s2, s3))
    return sat


def count_sat() -> int:
    """Count SAT scenarios without storing them (faster / less memory)."""
    count = 0
    for s1 in ALL_SCENES:
        for s2 in ALL_SCENES:
            for s3 in ALL_SCENES:
                if satisfies_K(s1, s2, s3):
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Fast clause-based count using inclusion-exclusion on per-scene predicate sets
# ---------------------------------------------------------------------------

def _predicate_sets():
    """Pre-compute which scenes satisfy each predicate."""
    sets = {k: [] for k in ["SameS","DiffS","TwoS","SameC","DiffC","TwoC"]}
    for i, sc in enumerate(ALL_SCENES):
        preds = scene_predicates(sc)
        for k, v in preds.items():
            if v:
                sets[k].append(i)
    return {k: set(v) for k, v in sets.items()}


def count_sat_fast() -> int:
    """
    Fast count using per-clause Cartesian product sizes (inclusion-exclusion
    handled by counting |A1|×|A2|×|A3| for each clause, then using union
    inclusion-exclusion over the 6 clauses).
    """
    pred_sets = _predicate_sets()

    clause_sets = [
        (pred_sets["DiffS"], pred_sets["DiffS"], pred_sets["DiffS"]),
        (pred_sets["TwoS"],  pred_sets["TwoS"],  pred_sets["TwoS"]),
        (pred_sets["SameS"], pred_sets["SameS"], pred_sets["SameS"]),
        (pred_sets["DiffC"], pred_sets["DiffC"], pred_sets["DiffC"]),
        (pred_sets["TwoC"],  pred_sets["TwoC"],  pred_sets["TwoC"]),
        (pred_sets["SameC"], pred_sets["SameC"], pred_sets["SameC"]),
    ]

    # Represent each scenario as a triple of scene indices.
    # A scenario is in clause C iff scene1-idx ∈ A, scene2-idx ∈ B, scene3-idx ∈ C.
    # |union of 6 clauses| via inclusion-exclusion over subsets.
    # Since clauses are independent Cartesian products we can use:
    #   |Ci| = |Ai| * |Bi| * |Di|
    # and for intersections:
    #   |Ci ∩ Cj| = |Ai∩Aj| * |Bi∩Bj| * |Di∩Dj|
    # This works because the predicates on different positions are independent.

    n = len(clause_sets)
    total = 0
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            # Intersect the per-position sets for all clauses in combo
            a = clause_sets[combo[0]][0]
            b = clause_sets[combo[0]][1]
            c = clause_sets[combo[0]][2]
            for idx in combo[1:]:
                a = a & clause_sets[idx][0]
                b = b & clause_sets[idx][1]
                c = c & clause_sets[idx][2]
            sign = (-1) ** (r + 1)
            total += sign * len(a) * len(b) * len(c)
    return total


# ---------------------------------------------------------------------------
# Random SAT scenario sampler (rejection sampling)
# ---------------------------------------------------------------------------

def sample_sat_scenarios(
    n: int,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[Scenario]:
    """
    Sample N SAT scenarios uniformly at random using rejection sampling.

    Parameters
    ----------
    n       : number of SAT scenarios to return
    seed    : optional random seed for reproducibility
    verbose : print progress

    Returns
    -------
    List of N SAT scenarios, each a tuple of 3 scenes.
    Each scene is ((s0,s1,s2), (c0,c1,c2)).
    """
    rng = random.Random(seed)
    results = []
    attempts = 0
    while len(results) < n:
        s1 = rng.choice(ALL_SCENES)
        s2 = rng.choice(ALL_SCENES)
        s3 = rng.choice(ALL_SCENES)
        attempts += 1
        if satisfies_K(s1, s2, s3):
            results.append((s1, s2, s3))
    if verbose:
        sat_rate = n / attempts
        print(f"Sampled {n} SAT scenarios in {attempts} attempts "
              f"(acceptance rate: {sat_rate:.2%})")
    return results


# ---------------------------------------------------------------------------
# Convert a scenario to a [3, 3, 2] numpy array
# ---------------------------------------------------------------------------

def scenario_to_array(scenario: Scenario) -> np.ndarray:
    """
    Convert a scenario (3 scenes) to a [3, 3, 2] numpy array.

    arr[i, j, 0] = shape of object j in scene i
    arr[i, j, 1] = color of object j in scene i
    """
    arr = np.zeros((3, 3, 2), dtype=np.int64)
    for i, scene in enumerate(scenario):
        shapes, colors = scene
        for j in range(3):
            arr[i, j, 0] = shapes[j]
            arr[i, j, 1] = colors[j]
    return arr


def print_scenario(scenario: Scenario, idx: int = 0):
    """Pretty-print a scenario."""
    arr = scenario_to_array(scenario)
    shape_names = {0: "circle", 1: "square", 2: "triangle"}
    color_names = {0: "red", 1: "green", 2: "blue"}
    print(f"\n--- Scenario {idx} ---")
    for i, scene in enumerate(scenario):
        shapes, colors = scene
        objs = [f"{shape_names[s]}/{color_names[c]}" for s, c in zip(shapes, colors)]
        preds = scene_predicates(scene)
        active = [k for k, v in preds.items() if v]
        print(f"  Scene {i+1}: {objs}  →  predicates: {active}")
    clauses = which_clauses(*scenario)
    print(f"  SAT via: {clauses}")
    print(f"  Array shape [3,3,2]:\n{arr}")


# ---------------------------------------------------------------------------
# Clause-level SAT count breakdown
# ---------------------------------------------------------------------------

def clause_breakdown():
    """
    Print count of scenarios satisfying each individual clause.
    (Scenarios may be counted multiple times if they satisfy >1 clause.)
    """
    pred_sets = _predicate_sets()
    clause_configs = [
        ("DiffS1∧DiffS2∧DiffS3", "DiffS", "DiffS", "DiffS"),
        ("TwoS1∧TwoS2∧TwoS3",    "TwoS",  "TwoS",  "TwoS"),
        ("SameS1∧SameS2∧SameS3", "SameS", "SameS", "SameS"),
        ("DiffC1∧DiffC2∧DiffC3", "DiffC", "DiffC", "DiffC"),
        ("TwoC1∧TwoC2∧TwoC3",    "TwoC",  "TwoC",  "TwoC"),
        ("SameC1∧SameC2∧SameC3", "SameC", "SameC", "SameC"),
    ]
    print("\nPer-clause SAT counts (may overlap):")
    print(f"  {'Clause':<30} | #scenes per pos | #scenarios")
    print(f"  {'-'*30}-+-{'-'*15}-+-{'-'*12}")
    for name, p1, p2, p3 in clause_configs:
        n1, n2, n3 = len(pred_sets[p1]), len(pred_sets[p2]), len(pred_sets[p3])
        n_scen = n1 * n2 * n3
        print(f"  {name:<30} | {n1:^15} | {n_scen:>12,}")

    total_with_overlap = sum(
        len(pred_sets[p1]) * len(pred_sets[p2]) * len(pred_sets[p3])
        for _, p1, p2, p3 in clause_configs
    )
    total_unique = count_sat_fast()
    print(f"\n  Sum with overlaps : {total_with_overlap:>12,}")
    print(f"  Unique SAT (exact): {total_unique:>12,}")
    print(f"  Total universe    : {len(ALL_SCENES)**3:>12,}")
    print(f"  SAT fraction      : {total_unique / len(ALL_SCENES)**3:.4%}")


# ---------------------------------------------------------------------------
# PyTorch save / load
# ---------------------------------------------------------------------------

def scenarios_to_tensor(scenarios: List[Scenario]) -> torch.Tensor:
    """
    Convert a list of N scenarios into a single LongTensor of shape [N, 3, 3, 2].

      tensor[n, i, j, 0] = shape of object j in scene i of scenario n
      tensor[n, i, j, 1] = color of object j in scene i of scenario n

    Values are in {0, 1, 2}.
    """
    arr = np.stack([scenario_to_array(sc) for sc in scenarios], axis=0)  # [N,3,3,2]
    return torch.from_numpy(arr).long()


def save_scenarios_pt(
    scenarios: List[Scenario],
    path: str,
    include_labels: bool = True,
) -> torch.Tensor:
    """
    Save N scenarios to a .pt file.

    Stored dict keys
    ----------------
    'data'    : LongTensor [N, 3, 3, 2]  — the scenario arrays
                  data[:, :, :, 0]  shapes  (values in {0,1,2})
                  data[:, :, :, 1]  colors  (values in {0,1,2})
    'labels'  : LongTensor [N]            — all 1 (every stored scenario is SAT)
    'clauses' : list[list[str]]           — which K-clauses each scenario satisfies

    Parameters
    ----------
    scenarios      : list of Scenario tuples (output of sample_sat_scenarios)
    path           : destination file path, e.g. "data/sat_1000.pt"
    include_labels : if True, store a 'labels' tensor of ones alongside 'data'

    Returns
    -------
    The [N, 3, 3, 2] data tensor (also saved to disk).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    data = scenarios_to_tensor(scenarios)                    # [N, 3, 3, 2]
    clauses = [which_clauses(*sc) for sc in scenarios]

    payload = {"data": data, "clauses": clauses}
    if include_labels:
        payload["labels"] = torch.ones(len(scenarios), dtype=torch.long)

    torch.save(payload, path)
    print(f"Saved {len(scenarios)} scenarios → {path}  (tensor shape: {list(data.shape)})")
    return data


def load_scenarios_pt(path: str):
    """
    Load a .pt file saved by save_scenarios_pt.

    Returns
    -------
    dict with keys 'data' (LongTensor [N,3,3,2]), optionally 'labels', 'clauses'.

    Quick access
    ------------
    d = load_scenarios_pt('sat_1000.pt')
    shapes = d['data'][:, :, :, 0]   # [N, 3, 3]
    colors = d['data'][:, :, :, 1]   # [N, 3, 3]
    """
    payload = torch.load(path, weights_only=False)
    data = payload["data"]
    print(f"Loaded {data.shape[0]} scenarios from {path}  (tensor shape: {list(data.shape)})")
    return payload

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kandinsky SAT scenario generator"
    )
    parser.add_argument("--enumerate", action="store_true",
                        help="Enumerate and print all SAT scenarios (slow!)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample N random SAT scenarios")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--count", action="store_true",
                        help="Print the total number of SAT scenarios")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save sampled scenarios as .pt (requires --sample)")
    args = parser.parse_args()

    print("=== Kandinsky SAT Analysis ===")
    print(f"  Shapes: {{0,1,2}}  |  Colors: {{0,1,2}}")
    print(f"  Total scenes in universe: {len(ALL_SCENES)}")

    clause_breakdown()

    if args.count or not (args.enumerate or args.sample):
        n = count_sat_fast()
        print(f"\nTotal SAT scenarios: {n:,} / {len(ALL_SCENES)**3:,}")

    if args.enumerate:
        print("\nEnumerating all SAT scenarios (this may take a few minutes)...")
        scenarios = enumerate_all_sat()
        print(f"Found {len(scenarios):,} SAT scenarios.")
        for i, scen in enumerate(scenarios[:20]):
            print_scenario(scen, idx=i)
        if len(scenarios) > 20:
            print(f"  ... ({len(scenarios) - 20} more)")

    if args.sample > 0:
        print(f"\nSampling {args.sample} random SAT scenarios...")
        scenarios = sample_sat_scenarios(args.sample, seed=args.seed)
        for i, scen in enumerate(scenarios):
            print_scenario(scen, idx=i)

        # Also demonstrate the numpy array form
        print("\nFirst scenario as [3,3,2] numpy array:")
        arr = scenario_to_array(scenarios[0])
        print("  arr[:,:,0]  (shapes):\n  ", arr[:, :, 0])
        print("  arr[:,:,1]  (colors):\n  ", arr[:, :, 1])

        if args.output:
            save_scenarios_pt(scenarios, args.output)



# ---------------------------------------------------------------------------
# kand_label — reference implementation (mirrors the validation function)
# ---------------------------------------------------------------------------

def _row_predicate(row: torch.Tensor) -> dict:
    """
    Compute same/diff/two predicates for a batch of 3-element rows.

    row : (B, 3)  integer tensor
    Returns dict with keys 'same', 'diff', 'two' — each a (B,) bool tensor.
    """
    a, b, c = row[:, 0], row[:, 1], row[:, 2]
    same = (a == b) & (b == c)
    diff = (a != b) & (b != c) & (a != c)
    two  = ~same & ~diff
    return {"same": same, "diff": diff, "two": two}


def kand_label(concepts: torch.Tensor) -> torch.Tensor:
    """
    Compute the ground-truth Kand-Logic label for a batch of scenarios.

    concepts : (B, 3, 3, 2)
                 concepts[b, row, col, 0] = shape  of object col in scene row
                 concepts[b, row, col, 1] = color  of object col in scene row

    Returns  : (B,) bool tensor — True = SAT (positive).

    K <=>  (DiffS1 & DiffS2 & DiffS3) | (TwoS1 & TwoS2 & TwoS3) | (SameS1 & SameS2 & SameS3)
        |  (DiffC1 & DiffC2 & DiffC3) | (TwoC1 & TwoC2 & TwoC3) | (SameC1 & SameC2 & SameC3)
    """
    shapes = concepts[:, :, :, 0]   # (B, 3, 3)
    colors = concepts[:, :, :, 1]   # (B, 3, 3)

    sp = [_row_predicate(shapes[:, r, :]) for r in range(3)]
    cp = [_row_predicate(colors[:, r, :]) for r in range(3)]

    diff_s = sp[0]["diff"] & sp[1]["diff"] & sp[2]["diff"]
    two_s  = sp[0]["two"]  & sp[1]["two"]  & sp[2]["two"]
    same_s = sp[0]["same"] & sp[1]["same"] & sp[2]["same"]

    diff_c = cp[0]["diff"] & cp[1]["diff"] & cp[2]["diff"]
    two_c  = cp[0]["two"]  & cp[1]["two"]  & cp[2]["two"]
    same_c = cp[0]["same"] & cp[1]["same"] & cp[2]["same"]

    return diff_s | two_s | same_s | diff_c | two_c | same_c


def validate_with_kand_label(path: str) -> None:
    """
    Load a saved .pt file and confirm every stored scenario passes kand_label.

    Raises AssertionError if any scenario fails — prints a full report otherwise.
    """
    payload = load_scenarios_pt(path)
    data    = payload["data"].long()
    labels  = kand_label(data)

    n_total = len(labels)
    n_pass  = labels.sum().item()
    n_fail  = n_total - n_pass

    print(f"\n  kand_label validation on '{path}':")
    print(f"    Total scenarios : {n_total}")
    print(f"    SAT (pass)      : {n_pass}")
    print(f"    FAIL            : {n_fail}")

    if n_fail > 0:
        fail_idx = (~labels).nonzero(as_tuple=True)[0].tolist()
        print(f"    Failing indices : {fail_idx[:20]}{'...' if len(fail_idx) > 20 else ''}")
        raise AssertionError(f"{n_fail} scenario(s) failed kand_label — sampler/label mismatch!")

    print("    All scenarios pass kand_label — sampler and label are consistent.")


# ---------------------------------------------------------------------------
# UNSAT scenario sampler
# ---------------------------------------------------------------------------

def sample_unsat_scenarios(
    n: int,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[Scenario]:
    """
    Sample N UNSAT scenarios uniformly at random using rejection sampling.

    UNSAT means the triple (s1, s2, s3) does NOT satisfy K — none of the
    six clauses holds across all three scenes.

    ~48% of the universe is UNSAT, so acceptance rate mirrors the SAT sampler.

    Parameters
    ----------
    n       : number of UNSAT scenarios to return
    seed    : optional random seed for reproducibility
    verbose : print acceptance stats

    Returns
    -------
    List of N UNSAT scenarios, each a tuple of 3 scenes ((shapes), (colors)).
    """
    rng = random.Random(seed)
    results = []
    attempts = 0
    while len(results) < n:
        s1 = rng.choice(ALL_SCENES)
        s2 = rng.choice(ALL_SCENES)
        s3 = rng.choice(ALL_SCENES)
        attempts += 1
        if not satisfies_K(s1, s2, s3):
            results.append((s1, s2, s3))
    if verbose:
        print(f"Sampled {n} UNSAT scenarios in {attempts} attempts "
              f"(acceptance rate: {n/attempts:.2%})")
    return results


def save_paired_pt(
    n: int,
    path_sat: str,
    path_unsat: str,
    seed: Optional[int] = None,
) -> None:
    """
    Generate and save N SAT and N UNSAT scenarios as two paired .pt files.

    Useful for training binary classifiers — both files have the same
    structure as save_scenarios_pt output, with 'labels' = 1 (SAT) or 0 (UNSAT).

    Parameters
    ----------
    n          : number of scenarios per class
    path_sat   : output path for the SAT tensor, e.g. "data/sat_1000.pt"
    path_unsat : output path for the UNSAT tensor, e.g. "data/unsat_1000.pt"
    seed       : shared base seed (UNSAT uses seed+1 to stay independent)
    """
    sat_scenarios   = sample_sat_scenarios(n,   seed=seed,            verbose=True)
    unsat_scenarios = sample_unsat_scenarios(n, seed=None if seed is None else seed + 1,
                                             verbose=True)

    sat_data   = scenarios_to_tensor(sat_scenarios)
    unsat_data = scenarios_to_tensor(unsat_scenarios)

    Path(path_sat).parent.mkdir(parents=True, exist_ok=True)
    Path(path_unsat).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "data":    sat_data,
        "labels":  torch.ones(n, dtype=torch.long),
        "clauses": [which_clauses(*sc) for sc in sat_scenarios],
    }, path_sat)
    print(f"Saved {n} SAT   scenarios → {path_sat}   {list(sat_data.shape)}")

    torch.save({
        "data":    unsat_data,
        "labels":  torch.zeros(n, dtype=torch.long),
        "clauses": [[] for _ in unsat_scenarios],
    }, path_unsat)
    print(f"Saved {n} UNSAT scenarios → {path_unsat}  {list(unsat_data.shape)}")

if __name__ == '__main__':
    seed = 42
    number_of_samples = 1
    save_paired_pt(number_of_samples, f'src/datasets/KandLogic/sat_{number_of_samples}.pt', f'src/datasets/KandLogic/unsat_{number_of_samples}.pt', seed)