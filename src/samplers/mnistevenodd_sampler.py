from z3 import *
import torch

TRAIN_PAIRS = [
    (0, 6), (2, 8), (4, 6), (4, 8), (6, 0), (8, 2), (6, 4), (8, 4),  # even+even
    (1, 5), (3, 7), (1, 9), (3, 9), (5, 1), (7, 3), (9, 1), (9, 3),  # odd+odd
]

class Sampler:
    def __init__(self):
        self.pairs_cache = self.get_pairs_cache()
        self.evenodd_cache = self._build_evenodd_cache()
        self.ood_cache = self._build_ood_cache()
        self.tensorized_pairs_cache = self.tensorized_cache()

    def _build_ood_cache(self):
        """
        Mixed-parity pairs (even+odd / odd+even) that are NOT in TRAIN_PAIRS,
        i.e. the out-of-distribution test support.
        Layout: [n, d1, d2] — identical to evenodd_cache.
        """
        train_set = set(TRAIN_PAIRS)
        rows = []
        for n in range(19):
            for (d1, d2) in self.pairs_cache[n]:
                is_mixed_parity = (d1 % 2) != (d2 % 2)
                not_in_train    = (d1, d2) not in train_set
                if is_mixed_parity and not_in_train:
                    rows.append([n, d1, d2])
        if not rows:
            return torch.zeros((0, 3), dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long)
           
    def _build_evenodd_cache(self):
        """
        For each sum n, store only the TRAIN_PAIRS (restricted support) that
        produce that sum. Layout identical to tensorized_pairs_cache: [n, d1, d2].
        """
        pair_set = set(TRAIN_PAIRS)
        rows = []
        for n in range(19):
            for (d1, d2) in self.pairs_cache[n]:   # pairs_cache already has all valid digit pairs
                if (d1, d2) in pair_set:
                    rows.append([n, d1, d2])
        if not rows:
            return torch.zeros((0, 3), dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long)

    def addition_solver(self, n):
        n_val = int(n)

        s = Solver()
        d1, d2 = Int('d1'), Int('d2')
        s.add(d1 >= 0, d1 <= 9, d2 >= 0, d2 <= 9)
        s.add(d1 + d2 == n_val)

        return s, d1, d2

    def get_pairs_cache(self):
        z3_cache = {}
        for n in range(19):
            s, d1, d2 = self.addition_solver(n)

            solutions = []
            while s.check() == sat:
                m = s.model()
                sol = (m[d1].as_long(), m[d2].as_long()) #type:ignore
                solutions.append(sol)
                # Block this solution to find the next
                s.add(Or(d1 != sol[0], d2 != sol[1]))
            z3_cache[int(n)] = solutions
        return z3_cache

    def tensorized_cache(self):
        samples = []

        for n in range(19):
            pairs = torch.tensor(self.pairs_cache[n], dtype=torch.long)

            # column containing n
            key = torch.full((pairs.shape[0], 1), n, dtype=torch.long)

            # concatenate as [n, a, b]
            samples.append(torch.cat([key, pairs], dim=1))

        return torch.cat(samples, dim=0)

    def batch_sample(self, n_tensor):
        device = n_tensor.device
        samples = self.tensorized_pairs_cache.to(device)
        cache_n = samples[:, 0]

        mask = n_tensor[:, None] == cache_n[None, :]
        rand = torch.rand(mask.shape, device=device)
        rand[~mask] = -1.0

        idx = rand.argmax(dim=1)

        true = (samples[idx, 1:].sum(dim=1) == n_tensor).any()
        assert true.item() == True

        # return only pairs (a,b)
        return samples[idx, 1:]

    def batch_k_sample(self, n_tensor, K=1):
        """
        Sample K pairs per batch element, restricted to TRAIN_PAIRS.
        Uses evenodd_cache instead of the full tensorized_pairs_cache.
        """
        device  = n_tensor.device
        samples = self.evenodd_cache.to(device)    # [M, 3]  ← was tensorized_pairs_cache
        cache_n = samples[:, 0]

        mask         = n_tensor[:, None] == cache_n[None, :]   # [B, M]
        mask_expanded = mask[:, None, :].expand(-1, K, -1)     # [B, K, M]

        rand = torch.rand(mask_expanded.shape, device=device)
        rand[~mask_expanded] = -1.0

        idx = rand.argmax(dim=2)                               # [B, K]
        return samples[idx, 1:]                                # [B, K, 2]

    def batch_k_sample_ood(self, n_tensor, K=1):
        """
        Sample K out-of-distribution (mixed-parity) pairs per batch element.
        Raises if any requested sum has no OOD pairs in the cache.

        Args:
            n_tensor: [B,]  target sums
            K:        int   number of pairs to sample per element

        Returns:
            [B, K, 2] pairs drawn exclusively from the OOD support
        """
        device  = n_tensor.device
        samples = self.ood_cache.to(device)   # [M, 3]
        cache_n = samples[:, 0]

        mask = n_tensor[:, None] == cache_n[None, :]   # [B, M]
        assert mask.any(dim=1).all(), \
            f"Sums with no OOD pairs: {n_tensor[~mask.any(dim=1)].tolist()}"

        mask_expanded = mask[:, None, :].expand(-1, K, -1)   # [B, K, M]
        rand = torch.rand(mask_expanded.shape, device=device)
        rand[~mask_expanded] = -1.0

        idx = rand.argmax(dim=2)              # [B, K]
        return samples[idx, 1:]               # [B, K, 2]
    
    def walk(self, samples, addition_labels):
        """
        Walk within the restricted EvenOdd support.
        For each sample in [B, K, 2], pick a *different* valid TRAIN_PAIR
        that shares the same sum.  If no alternative exists for a given sum,
        the original pair is kept.

        Args:
            samples:          [B, K, 2]  current (d1, d2) pairs
            addition_labels:  [B,]       the target sum for each item in the batch

        Returns:
            candidates:       [B, K, 2]  neighbour pairs within restricted support
        """
        device   = samples.device
        B, K, _  = samples.shape

        cache    = self.evenodd_cache.to(device)   # [M, 3]  columns: [n, d1, d2]
        cache_n  = cache[:, 0]                     # [M,]
        cache_p  = cache[:, 1:]                    # [M, 2]

        # Expand addition_labels to [B, K] then flatten to [B*K,] for vectorised lookup
        labels_bk = addition_labels[:, None].expand(-1, K).reshape(-1)   # [B*K,]
        cur_bk    = samples.reshape(-1, 2)                                # [B*K, 2]

        # For each (b,k) entry find all cache rows with matching sum
        sum_mask  = labels_bk[:, None] == cache_n[None, :]                # [B*K, M]

        # Exclude the current pair so we always move to a *different* neighbour
        same_pair = (cur_bk[:, None, :] == cache_p[None, :, :]).all(-1)  # [B*K, M]
        move_mask = sum_mask & ~same_pair                                  # [B*K, M]

        # Random selection among valid neighbours; fall back to current if none exist
        rand      = torch.rand(move_mask.shape, device=device)
        rand[~move_mask] = -1.0
        best_idx  = rand.argmax(dim=1)                                    # [B*K,]

        has_neighbour = move_mask.any(dim=1)                              # [B*K,]
        chosen_pair   = cache_p[best_idx]                                 # [B*K, 2]
        result        = torch.where(
            has_neighbour[:, None].expand(-1, 2),
            chosen_pair,
            cur_bk
        )

        return result.reshape(B, K, 2)

    def batch_k_sample_full(self, n_tensor, K=1):
        """
        Sample K pairs per batch element from the FULL digit space
        satisfying d1 + d2 = n.

        Uses tensorized_pairs_cache instead of evenodd_cache.

        Args:
            n_tensor: [B,]
            K: int

        Returns:
            [B, K, 2]
        """
        device  = n_tensor.device
        samples = self.tensorized_pairs_cache.to(device)   # [M, 3]
        cache_n = samples[:, 0]

        # valid rows for each sum
        mask = n_tensor[:, None] == cache_n[None, :]       # [B, M]

        # expand for K samples
        mask_expanded = mask[:, None, :].expand(-1, K, -1) # [B, K, M]

        # random selection
        rand = torch.rand(mask_expanded.shape, device=device)
        rand[~mask_expanded] = -1.0

        idx = rand.argmax(dim=2)                           # [B, K]

        return samples[idx, 1:]                            # [B, K, 2]

    def walk_full(self, samples, addition_labels):
        """
        Walk within the FULL digit-pair support.

        For each sample in [B, K, 2], pick a *different*
        valid pair sharing the same sum.

        If no alternative exists (only possible for sums 0 or 18),
        the original pair is kept.

        Args:
            samples:         [B, K, 2]
            addition_labels: [B]

        Returns:
            [B, K, 2]
        """
        device   = samples.device
        B, K, _  = samples.shape

        cache    = self.tensorized_pairs_cache.to(device)  # [M, 3]
        cache_n  = cache[:, 0]
        cache_p  = cache[:, 1:]                            # [M, 2]

        # flatten batch/k dimension
        labels_bk = addition_labels[:, None].expand(-1, K).reshape(-1)
        cur_bk    = samples.reshape(-1, 2)

        # same-sum mask
        sum_mask = labels_bk[:, None] == cache_n[None, :]  # [B*K, M]

        # exclude current pair
        same_pair = (cur_bk[:, None, :] == cache_p[None, :, :]).all(-1)

        move_mask = sum_mask & ~same_pair

        # random neighbour selection
        rand = torch.rand(move_mask.shape, device=device)
        rand[~move_mask] = -1.0

        best_idx = rand.argmax(dim=1)

        has_neighbour = move_mask.any(dim=1)

        chosen_pair = cache_p[best_idx]

        result = torch.where(
            has_neighbour[:, None].expand(-1, 2),
            chosen_pair,
            cur_bk
        )
        return result.reshape(B, K, 2)
    
if __name__ == "__main__":
    sampler = Sampler()

    n_values = torch.tensor([6, 10, 12, 12])
    samples = sampler.batch_sample(n_values)
    #print(samples.shape)

    k_samples = sampler.batch_k_sample(n_values, K=2)
    print(k_samples)

    k_samples_odd = sampler.batch_k_sample_ood(n_values, K=2)
    print(k_samples_odd)

    candidates = sampler.walk(k_samples, n_values)
    print(candidates)