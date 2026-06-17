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

class PrimitivesLeNet(nn.Module):
    """
    LeNet encoder with MLP classifier head.

    """

    def __init__(
        self,
        x_dim=3,
        hid_dim=64,
        z_dim=64,
        num_classes=10,
        mlp_hidden=256,
    ):
        super().__init__()


        self.encoder = nn.Sequential(
            conv_block(x_dim,   hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

        self.emb_dim = z_dim

        if num_classes is None:
            raise ValueError(
                "num_classes must be specified in classifier mode."
            )

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, num_classes),
        )

    def encode(self, x):
        z = self.encoder(x)
        return z.view(z.size(0), -1)

    def forward(self, x):
        z = self.encode(x)
        return self.classifier(z)