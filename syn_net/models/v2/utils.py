from typing import Callable, Optional, Union

import numpy as np
from sklearn.neighbors import BallTree, KDTree
from torch import Tensor
import torch
from torch.nn import functional as F

Tree = Union[KDTree, BallTree]


def get_loss(loss: str) -> Callable[[Tensor, Tensor], Tensor]:
    if loss == "cross_entropy":
        return F.cross_entropy
    if loss == "mse":
        return F.mse_loss
    if loss == "l1":
        return F.l1_loss
    if loss == "huber":
        return F.huber_loss

    raise ValueError(f"Unsupported loss function! got: {loss}")


def get_metric(metric: str) -> Callable[[Tensor, Tensor, Tree], Tensor]:
    if metric == "cross_entropy":
        return cross_entropy
    if metric == "accuracy":
        return accuracy
    if metric == "nn_accuracy":
        return nn_accuracy
    if metric == "mse":
        return mse
    if metric == "l1":
        return l1
    if metric == "huber":
        return huber

    raise ValueError(f"Unsupported validation metric! got: {metric}")


def cross_entropy(y_hat: Tensor, y: Tensor, _):
    return F.cross_entropy(y_hat, y)


def accuracy(y_hat: Tensor, y: Tensor, _):
    return 1 - (y_hat.argmax(1) == y).sum() / len(y)


def nn_accuracy(y_hat: Tensor, y: Tensor, tree):
    y = nn_search_list(y.detach().cpu().numpy(), tree)
    y_hat = nn_search_list(y_hat.detach().cpu().numpy(), tree)

    return torch.tensor(1 - (y_hat == y).sum() / len(y))


def mse(y_hat: Tensor, y: Tensor, _):
    return F.mse_loss(y_hat, y)


def l1(y_hat: Tensor, y: Tensor, _):
    return F.l1_loss(y_hat, y)


def huber(y_hat: Tensor, y: Tensor, _):
    return F.huber_loss(y_hat, y)


def nn_search_list(y, tree: Tree):
    return np.array(
        [tree.query(emb.reshape(1, -1), 1, return_distance=False)[0][0] for emb in y]
    )


def build_tree(out_feat: str) -> Optional[BallTree]:
    if out_feat == "gin":
        X = np.load("/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_gin.npy")
        return BallTree(X, metric="euclidean")

    if out_feat == "fp_4096":
        X = np.load("/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_4096.npy")
        return BallTree(X, metric="euclidean")

    if out_feat == "fp_256":
        X = np.load("/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_256.npy")
        return BallTree(X, metric=cosine_distance)

    if out_feat == "rdkit2d":
        X = np.load("/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_rdkit2d.npy")
        return BallTree(X, metric="euclidean")

    return None


def cosine_distance(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-15):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2) + eps)
