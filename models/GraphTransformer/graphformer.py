import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .ViT import *
from .gcn import GCNBlock

from torch.nn import Linear
from typing import Optional, Tuple

from torch import Tensor


def dense_mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum("ijk->ij", adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
        dim=(-1, -2),
    )
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum("ijj->i", x)


def _rank3_diag(x: Tensor) -> Tensor:
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))

    return out


class Classifier(nn.Module):
    def __init__(self, n_class=7, n_features: int = 512, loss_func=None):
        super(Classifier, self).__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100

        self.transformer = VisionTransformer(
            num_classes=n_class, embed_dim=self.embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.criterion = loss_func

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(
            n_features,
            self.embed_dim,
            self.bn,
            self.add_self,
            self.normalize_embedding,
            0.0,
            0,
        )  # 64->128
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)  # 100-> 20

    def forward(self, node_feat, labels, adj, mask):

        X = node_feat
        X = mask.unsqueeze(2) * X
        X = self.conv1(X, adj, mask)
        s = self.pool1(X)
        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)
        out = self.transformer(X)
        # loss
        loss = self.criterion(out, labels)
        loss = loss + mc1 + o1
        # pred
        return out, labels, loss
