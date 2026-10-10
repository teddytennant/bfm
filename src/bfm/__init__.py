"""Bernoulli flow models, linear schedule only."""

from bfm._host import ensure_nix_ld

ensure_nix_ld()

from bfm.path import alpha, g, gamma, time_grid
from bfm.posterior import (
    forward_flip,
    posterior_prob,
    reverse_bernoulli_prob,
)
from bfm.train import batch_loss, bce_loss, init_mlp, predict_x0, sample, train_step

__all__ = [
    "alpha",
    "batch_loss",
    "bce_loss",
    "ensure_nix_ld",
    "forward_flip",
    "g",
    "gamma",
    "init_mlp",
    "posterior_prob",
    "predict_x0",
    "reverse_bernoulli_prob",
    "sample",
    "time_grid",
    "train_step",
]

__version__ = "0.1.0"
