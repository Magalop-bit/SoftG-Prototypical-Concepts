import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════
# Backbone
# ═══════════════════════════════════════════════════════════════════════════

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class PrimitivesProtoNet(nn.Module):
    """
    ProtoNet encoder with optional MLP classifier head.

    mode="proto"      -> returns embeddings
    mode="classifier" -> returns class logits
    """

    def __init__(
        self,
        x_dim=3,
        hid_dim=64,
        z_dim=64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(x_dim,   hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

        self.emb_dim = z_dim

    def encode(self, x):
        z = self.encoder(x)
        return z.view(z.size(0), -1)

    def forward(self, x):
        z = self.encode(x)
        return z



def euclidean_dist(x, y):
    """
    Squared Euclidean distance:
    (N,D) x (M,D) -> (N,M)
    """
    n, d = x.shape
    m = y.shape[0]

    return (
        x.unsqueeze(1).expand(n, m, d)
        - y.unsqueeze(0).expand(n, m, d)
    ).pow(2).sum(2)


def prototypical_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_support: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard prototypical-networks episode loss.

    embeddings : (N, D)  – encoder output for support+query, interleaved by
                           the label ordering assumed by PrototypicalLoss
    labels     : (N,)    – integer class ids

    The function computes prototypes from the first n_support embeddings per
    class (in label order) and classifies the remaining ones.

    Returns (loss, accuracy).
    """
    device = embeddings.device
    #labels = labels[:,0] * 10 + labels[:,1]
    classes = torch.unique(labels)
    n_classes = len(classes)

    # --- prototypes ---
    support_proto = []
    query_emb_list = []
    query_lbl_list = []

    for i, c in enumerate(classes):
        mask = labels == c
        idxs = mask.nonzero(as_tuple=True)[0]
        support_proto.append(embeddings[idxs[:n_support]].mean(0))
        if len(idxs) > n_support:
            query_emb_list.append(embeddings[idxs[n_support:]])
            query_lbl_list.append(
                torch.full((len(idxs) - n_support,), i, dtype=torch.long, device=device)
            )

    if not query_emb_list:
        # No query samples – return zero loss
        return torch.tensor(0.0, requires_grad=True), torch.tensor(1.0)

    prototypes = torch.stack(support_proto)        # (n_classes, D)
    query_emb  = torch.cat(query_emb_list, dim=0)  # (Q, D)
    query_lbl  = torch.cat(query_lbl_list, dim=0)  # (Q,)

    # --- squared Euclidean distances -> log-softmax -> NLL ---
    dists    = torch.cdist(query_emb, prototypes, p=2) ** 2  # (Q, n_classes)
    log_p_y  = F.log_softmax(-dists, dim=1)
    loss     = F.nll_loss(log_p_y, query_lbl)
    acc      = (log_p_y.argmax(1) == query_lbl).float().mean()

    return loss, acc

def compute_prototypes(
    support_imgs: torch.Tensor,
    support_lbls: torch.Tensor,   # flat joint labels from {0,1,2,...,8}
    encoder:      PrimitivesProtoNet,
) -> torch.Tensor:
    """
    Returns prototypes of shape (3, 3, D):
      proto[shape, color, :] = mean embedding of that primitive class.
    Absent classes stay at zero.
    """
    embs   = encoder(support_imgs)          # (N_s, D)
    D      = embs.size(1)
    device = embs.device

    proto = torch.zeros(3, 3, D, device=device)   # [shape, color, D]

    for lbl in torch.unique(support_lbls):
        c     = lbl.item()
        color = c % 3     # 0,1,2
        shape = c // 3      # 0,1,2
        mask  = support_lbls == c
        proto[shape, color] = embs[mask].mean(0)

    return proto   # (3, 3, D)                                

def digit_probs(
    embeddings: torch.Tensor,   # (B, D)
    prototypes: torch.Tensor,   # (3, 3, D)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      color_probs : (B, 3)
      shape_probs : (B, 3)
    by marginalising over the joint prototype grid.
    """
    B, D = embeddings.shape
    flat_proto = prototypes.view(9, D)              # (9, D)  row = shape*3 + color 

    dists      = torch.cdist(embeddings, flat_proto, p=2) ** 2   # (B, 9)
    joint      = F.softmax(-dists, dim=1).view(B, 3, 3)          # (B, shape, color)

    shape_probs = joint.sum(dim=2)   # (B, 3)  marginal over shape
    color_probs = joint.sum(dim=1)   # (B, 3)  marginal over color

    return shape_probs, color_probs
