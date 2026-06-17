import torch
import torch.nn.functional as F

from src.backbones.Kandlogic.protoencoder import PrimitivesProtoNet, compute_prototypes, digit_probs

def kand_nesy_loss(
    images:      torch.Tensor,   # (B, N, N, C, H, W)
    labels:      torch.Tensor,   # (B,)  scene-level: 1=positive, 0=negative
    encoder:     PrimitivesProtoNet,
    prototypes:  torch.Tensor,   # (3, 3, D)  from compute_prototypes
    device:      torch.device,
):
    """
    Prototype-grounded semantic loss for Kand-Logic.

    For each scene:
      1. Embed all 9 primitives and get (color_probs, shape_probs) via digit_probs.
      2. Compute the scene-level probability of the Kand label being True
         by evaluating each of the 6 clauses in differentiable form and
         taking their soft-OR (1 - product of (1 - clause_prob)).
      3. Weight the BCE loss by the prototype confidence: how close the
         embedding is to its nearest prototype center. This is the
         'right for the right reasons' grounding — uncertain embeddings
         contribute less to the symbolic constraint gradient.
    """
    B, N, _, C, H, W = images.shape

    flat_imgs  = images.view(B * N * N, C, H, W).to(device)
    embs       = encoder(flat_imgs)                            # (B*9, D)

    shape_probs, color_probs = digit_probs(embs, prototypes)   # (B*9, 3) each

    # ── prototype confidence weight ───────────────────────────────────────
    # distance to nearest prototype center → low distance = high confidence
    flat_proto = prototypes.view(9, -1)                        # (9, D)
    dists      = torch.cdist(embs, flat_proto, p=2) ** 2       # (B*9, 9)
    min_dist   = dists.min(dim=1).values                       # (B*9,)
    # confidence: 1 when on a prototype, decays with distance
    conf       = torch.exp(-min_dist / (min_dist.detach().mean() + 1e-8))
    conf       = conf.view(B, N * N).mean(dim=1)               # (B,)  per scene

    # ── reshape probs to (B, 3-rows, 3-cols, 3-classes) ──────────────────
    sp = shape_probs.view(B, N, N, 3)   # (B, row, col, shape_class)
    cp = color_probs.view(B, N, N, 3)   # (B, row, col, color_class)

    # ── soft row predicates ───────────────────────────────────────────────
    # For each row r and attribute (shape or color), compute:
    #   P(same)  = sum_c  P(col0=c)*P(col1=c)*P(col2=c)
    #   P(diff)  = sum_{a≠b≠c}  P(col0=a)*P(col1=b)*P(col2=c)   = 1 - P(not-all-diff)
    #            = actually easiest as: 1 - P(same) - P(two)
    #   P(two)   = 3 * sum_c P(col0=c)*P(col1=c)*(1-P(col2=c))  (symmetrized)

    def soft_predicates(p):
        # p: (B, 3-rows, 3-cols, 3-classes)
        p0, p1, p2 = p[:, :, 0, :], p[:, :, 1, :], p[:, :, 2, :]  # (B,3,3) each

        same = (p0 * p1 * p2).sum(dim=-1)          # (B, 3-rows)

        # P(two): exactly two share a class
        # = sum_c [ P(a=c)P(b=c)(1-P(c_=c)) ] * 3 positions
        two  = (
            (p0 * p1 * (1 - p2)).sum(dim=-1) +
            (p0 * p2 * (1 - p1)).sum(dim=-1) +
            (p1 * p2 * (1 - p0)).sum(dim=-1)
        )                                           # (B, 3-rows)  -- overcounts same, fix:
        two  = two - 3 * same                       # remove triple-overlap contribution
        two  = two.clamp(min=0)

        diff = (1 - same - two).clamp(min=0)        # (B, 3-rows)

        return same, diff, two

    s_same, s_diff, s_two = soft_predicates(sp)    # each (B, 3-rows)
    c_same, c_diff, c_two = soft_predicates(cp)

    # ── soft clause probabilities (must hold for ALL 3 rows) ─────────────
    # P(clause) = prod over rows  (uses log for numerical stability)
    def all_rows(x):
        return x.prod(dim=1)   # (B,)

    p_diff_s = all_rows(s_diff)
    p_two_s  = all_rows(s_two)
    p_same_s = all_rows(s_same)
    p_diff_c = all_rows(c_diff)
    p_two_c  = all_rows(c_two)
    p_same_c = all_rows(c_same)

    # ── soft-OR of 6 clauses: P(K=True) ──────────────────────────────────
    # 1 - prod(1 - clause_i)
    clauses = torch.stack([p_diff_s, p_two_s, p_same_s,
                           p_diff_c, p_two_c, p_same_c], dim=1)   # (B, 6)
    p_K     = 1 - (1 - clauses.clamp(1e-7, 1 - 1e-7)).prod(dim=1) # (B,)

    # ── prototype-weighted BCE ────────────────────────────────────────────
    y = labels.float().to(device)
    bce = F.binary_cross_entropy(p_K, y, reduction='none')  # (B,)

    proto_pull = min_dist.mean()  

    return (conf.detach() * bce).mean(), proto_pull