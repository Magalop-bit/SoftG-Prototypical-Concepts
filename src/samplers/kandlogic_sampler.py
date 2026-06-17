import torch
import numpy as np
import random
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sampler:

    def __init__(self) -> None:
        self._POWERS = torch.tensor([243, 81, 27, 9, 3, 1], dtype=torch.long)
        self._MAX_SCENE_ID = 729   # 3^6

        self.sat_scenes_cache   = self._load_sat_scenes()
        self.unsat_scenes_cache = self._load_unsat_scenes()
        self.N = len(self.sat_scenes_cache)
        self._powers = self._POWERS.to(device)

        # BUG 1 FIX: build one index per cache — SAT and UNSAT rows are independent
        self._sat_lookup,   self._sat_hit_counts   = self._build_index(self.sat_scenes_cache)
        self._unsat_lookup, self._unsat_hit_counts = self._build_index(self.unsat_scenes_cache)

    # ------------------------------------------------------------------
    def _load_sat_scenes(self) -> torch.Tensor:
        path = 'src/datasets/Kandlogic/sat_200_000.pt'
        payload = torch.load(path, weights_only=False)
        return payload['data'].long().to(device)   # keep as long throughout

    def _load_unsat_scenes(self) -> torch.Tensor:
        path = 'src/datasets/Kandlogic/unsat_200_000.pt'
        payload = torch.load(path, weights_only=False)
        return payload['data'].long().to(device)

    # ------------------------------------------------------------------
    def _scene_ids(self, data: torch.Tensor) -> torch.Tensor:
        """[B, 3, 3, 2] → [B, 3]  integer scene IDs in [0, 729)."""
        ids = torch.zeros(data.shape[0], 3, dtype=torch.long, device=device)
        for pos in range(3):
            flat = torch.cat([data[:, pos, :, 0], data[:, pos, :, 1]], dim=1)  # [B, 6]
            ids[:, pos] = (flat * self._powers.unsqueeze(0)).sum(1)
        return ids

    def _build_index(self, cache: torch.Tensor):
        """
        Build a padded lookup table for `cache`.

          lookup     : [3, 729, max_hits]  — cache row indices per (pos, scene_id)
          hit_counts : [3, 729]            — number of valid entries per slot

        Slots with no hits are filled with -1.
        Called once for SAT cache and once for UNSAT cache.

        Fully vectorized — no Python loops over N.
        Old: O(N) .item() calls → ~25s at 200k. New: ~300ms at 200k.
        """
        N    = cache.shape[0]
        sids = self._scene_ids(cache).cpu()    # [N, 3], keep on CPU for scatter ops
        ones   = torch.ones(N, dtype=torch.long)
        arange = torch.arange(N, dtype=torch.long)

        # ── hit_counts via scatter_add ─────────────────────────────────────
        hit_counts = torch.zeros(3, self._MAX_SCENE_ID, dtype=torch.long)
        for pos in range(3):
            hit_counts[pos].scatter_add_(0, sids[:, pos], ones)

        max_hits = hit_counts.max().item()
        lookup   = torch.full((3, self._MAX_SCENE_ID, max_hits), -1, dtype=torch.long) #type:ignore

        # ── fill lookup: sort by scene_id, assign within-group slot index ──
        for pos in range(3):
            sorted_sids, order = sids[:, pos].sort(stable=True)

            # first_pos[sid] = first index in sorted array where sid appears
            first_pos = torch.full((self._MAX_SCENE_ID,), N, dtype=torch.long)
            first_pos.scatter_reduce_(0, sorted_sids, arange, reduce='amin')

            within = arange - first_pos[sorted_sids]      # slot within group
            lookup[pos, sorted_sids, within] = arange[order]

        return lookup.to(device), hit_counts.to(device)

    # ------------------------------------------------------------------
    def tensor_sat_sample_batch(self, batch_size: int, k: int = 1) -> torch.Tensor:
        """Sample [batch_size, k, 3, 3, 2] SAT scenarios uniformly from cache."""
        with torch.no_grad():
            indices = torch.randint(0, self.N, (batch_size, k), device=device)
            return self.sat_scenes_cache[indices]

    def tensor_unsat_sample_batch(self, batch_size: int, k: int = 1) -> torch.Tensor:
        """Sample [batch_size, k, 3, 3, 2] UNSAT scenarios uniformly from cache."""
        with torch.no_grad():
            indices = torch.randint(0, self.N, (batch_size, k), device=device)
            return self.unsat_scenes_cache[indices]

    def batch_k_sample(self, scene_labels: torch.Tensor,
                       batch_size: int, k: int = 1) -> torch.Tensor:
        scene_labels = scene_labels.to(device)

        sat_samples   = self.tensor_sat_sample_batch(batch_size, k)
        unsat_samples = self.tensor_unsat_sample_batch(batch_size, k)

        sat_mask   = (scene_labels == 1)
        unsat_mask = (scene_labels == 0)

        assert (sat_mask | unsat_mask).all(), \
            f"scene_labels must be 0 or 1, got: {scene_labels.unique().tolist()}"

        result = sat_samples.clone()
        result[unsat_mask] = unsat_samples[unsat_mask]
        return result

    # ------------------------------------------------------------------
    def _row_predicate(self, row: torch.Tensor) -> dict:
        """row : (B, 3) → dict of (B,) bool tensors for same/diff/two."""
        a, b, c = row[:, 0], row[:, 1], row[:, 2]
        same = (a == b) & (b == c)
        diff = (a != b) & (b != c) & (a != c)
        two  = ~same & ~diff
        return {"same": same, "diff": diff, "two": two}

    def kand_label(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        concepts : LongTensor (B, 3, 3, 2)
        Returns  : (B,) bool — True = SAT.
        """
        shapes = concepts[:, :, :, 0]
        colors = concepts[:, :, :, 1]

        sp = [self._row_predicate(shapes[:, r, :]) for r in range(3)]
        cp = [self._row_predicate(colors[:, r, :]) for r in range(3)]

        diff_s = sp[0]["diff"] & sp[1]["diff"] & sp[2]["diff"]
        two_s  = sp[0]["two"]  & sp[1]["two"]  & sp[2]["two"]
        same_s = sp[0]["same"] & sp[1]["same"] & sp[2]["same"]

        diff_c = cp[0]["diff"] & cp[1]["diff"] & cp[2]["diff"]
        two_c  = cp[0]["two"]  & cp[1]["two"]  & cp[2]["two"]
        same_c = cp[0]["same"] & cp[1]["same"] & cp[2]["same"]

        return diff_s | two_s | same_s | diff_c | two_c | same_c

    # ------------------------------------------------------------------
    def step(
        self,
        batch:     torch.Tensor,           # [B, 3, 3, 2] long
        cache:     torch.Tensor,           # [N, 3, 3, 2] long  — SAT or UNSAT
        lookup:    torch.Tensor,           # [3, 729, max_hits] — matching index
        hit_counts: torch.Tensor,          # [3, 729]           — matching counts
        fixed_pos: Optional[int] = None,
        generator: torch.Generator = None, #type:ignore
    ) -> torch.Tensor:
        """
        One projection step for a homogeneous batch (all SAT or all UNSAT).
        Caller is responsible for passing the correct (cache, lookup, hit_counts) pair.

        Returns LongTensor [B, 3, 3, 2].
        """
        B = batch.shape[0]

        fp = (torch.randint(0, 3, (B,), device=device, generator=generator)
              if fixed_pos is None
              else torch.full((B,), fixed_pos, dtype=torch.long, device=device))

        anchor     = batch[torch.arange(B, device=device), fp]            # [B, 3, 2]
        flat       = torch.cat([anchor[:, :, 0], anchor[:, :, 1]], dim=1) # [B, 6]
        anchor_ids = (flat * self._powers.unsqueeze(0)).sum(1)            # [B]

        n_hits   = hit_counts[fp, anchor_ids]                             # [B]
        rand_off = (torch.rand(B, device=device, generator=generator)
                    * n_hits.float()).long()
        rand_off = rand_off.clamp(0, lookup.shape[2] - 1)

        cache_rows = lookup[fp, anchor_ids, rand_off]                     # [B]

        miss = cache_rows < 0
        if miss.any():
            cache_rows[miss] = torch.randint(
                0, cache.shape[0], (miss.sum().item(),), device=device) #type:ignore

        return cache[cache_rows]   # [B, 3, 3, 2]  — dtype inherited from cache (long)

    def walk(
        self,
        batch:        torch.Tensor,   # [B, 3, 3, 2]
        scene_labels: torch.Tensor,   # [B]  — 1=SAT, 0=UNSAT
        n_steps:      int,
        fixed_pos:    Optional[int] = None,
        generator:    torch.Generator = None, #type:ignore
    ) -> torch.Tensor:
        """
        Walk each scenario for n_steps, respecting its label:
          SAT   scenarios walk within the SAT   cache
          UNSAT scenarios walk within the UNSAT cache

        Returns LongTensor [B, 3, 3, 2].
        """
        scene_labels = scene_labels.to(device)
        current      = batch.long().to(device)   # BUG 2 FIX: ensure long from the start

        sat_mask   = scene_labels == 1   # [B]
        unsat_mask = scene_labels == 0

        assert (sat_mask | unsat_mask).all(), \
            f"scene_labels must be 0 or 1, got: {scene_labels.unique().tolist()}"

        current_sat   = current[sat_mask]    # [B_sat,   3, 3, 2]
        current_unsat = current[unsat_mask]  # [B_unsat, 3, 3, 2]

        for _ in range(n_steps):
            if current_sat.shape[0] > 0:
                current_sat = self.step(
                    current_sat,
                    self.sat_scenes_cache,
                    self._sat_lookup,        # BUG 1 FIX: use SAT index
                    self._sat_hit_counts,
                    fixed_pos=fixed_pos,
                    generator=generator,
                )
            if current_unsat.shape[0] > 0:
                current_unsat = self.step(
                    current_unsat,
                    self.unsat_scenes_cache,
                    self._unsat_lookup,      # BUG 1 FIX: use UNSAT index
                    self._unsat_hit_counts,
                    fixed_pos=fixed_pos,
                    generator=generator,
                )

        # BUG 2 FIX: allocate with dtype=torch.long, not the default float32
        result = torch.empty_like(current)
        result[sat_mask]   = current_sat
        result[unsat_mask] = current_unsat
        return result


if __name__ == '__main__':
    sampler = Sampler()

    # Basic sampling
    sample = sampler.tensor_sat_sample_batch(batch_size=32, k=10)
    print(f"sat sample shape : {sample.shape}  dtype: {sample.dtype}")

    scene_labels = torch.tensor([1, 0, 1, 0]).to(device)
    scene_sample = sampler.batch_k_sample(scene_labels, batch_size=4, k=2)
    print(f"batch_k_sample   : {scene_sample.shape}")

    flat  = scene_sample.reshape(8, 3, 3, 2)
    check = sampler.kand_label(flat)
    print(f"kand_label       : {check}")

    # Walk — mixed SAT/UNSAT batch
    exp_labels = scene_labels[:, None].expand(-1, 2).reshape(8)
    walked = sampler.walk(flat, exp_labels, n_steps=1)
    print(f"walked shape     : {walked.shape}  dtype: {walked.dtype}")

    # Verify labels are preserved
    walked_labels = sampler.kand_label(walked)
    print(f"labels preserved : {(walked_labels == exp_labels.bool()).all().item()}")