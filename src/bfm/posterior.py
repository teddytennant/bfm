"""Forward flip kernel and exact reverse posterior for the linear schedule."""

import jax
import jax.numpy as jnp

from bfm.path import alpha, g, gamma


def reverse_bernoulli_prob(x_cur, alpha_prev, gam):
    """P(x_prev = 1 | x_cur, x0) with a supplied flip probability.

    p = g(x_cur, gamma) * alpha_prev / g(g(x_cur, gamma), 1 - alpha_prev).
    Exact for a coarse jump when gam is the coarse-gap gamma, not a one-step gamma.
    """
    mixed = g(x_cur, gam)
    num = mixed * alpha_prev
    den = g(mixed, 1.0 - alpha_prev)
    return num / den


def posterior_prob(x_cur, x0, t_prev, t_cur):
    """Reverse posterior from noisier (x_cur, t_cur) to cleaner time t_prev.

    t_prev > t_cur. alpha_prev = g(x0, (1 - t_prev) / 2) and gamma uses that
    same pair of times, including when the gap skips grid steps.
    """
    return reverse_bernoulli_prob(
        x_cur, alpha(x0, t_prev), gamma(t_prev, t_cur)
    )


def forward_probs(x_prev, t_prev, t_cur):
    """P(x_cur = 1 | x_prev) = g(x_prev, gamma) for the cleaner-to-noisier flip."""
    return g(x_prev, gamma(t_prev, t_cur))


def forward_flip(key, x_prev, t_prev, t_cur):
    """Sample the forward kernel: Bernoulli(g(x_prev, gamma))."""
    probs = forward_probs(x_prev, t_prev, t_cur)
    return jax.random.bernoulli(key, probs).astype(jnp.float32)
