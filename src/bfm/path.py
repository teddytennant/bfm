"""Linear Bernoulli path.

g(a, b) = a * (1 - b) + b * (1 - a) is a bit-flip mixture: a Bernoulli(a)
bit flipped with probability b is Bernoulli(g(a, b)).

Time runs from data to noise on t_tau = 1 - tau / T. t = 1 is data, t = 0 is noise.
The marginal is alpha_t = g(x0, (1 - t) / 2), so alpha(t=1) = x0 and alpha(t=0) = 0.5.
Only this linear schedule is implemented.
"""

import jax.numpy as jnp


def g(a, b):
    """Binary mixture g(a, b) = a * (1 - b) + b * (1 - a)."""
    return a * (1.0 - b) + b * (1.0 - a)


def time_grid(num_steps: int):
    """t_tau = 1 - tau / T for tau = 0..T. Index 0 is data, index T is noise."""
    steps = jnp.asarray(num_steps, dtype=jnp.float32)
    tau = jnp.arange(num_steps + 1, dtype=jnp.float32)
    return 1.0 - tau / steps


def alpha(x0, t):
    """Marginal Bernoulli parameter alpha_t = g(x0, (1 - t) / 2)."""
    return g(x0, (1.0 - t) / 2.0)


def gamma(t_prev, t_cur):
    """Flip probability from cleaner t_prev to noisier t_cur.

    t_prev > t_cur. gamma = 0.5 * (t_prev - t_cur) / t_prev.
    The gap may span several grid steps. That coarse gamma is the exact
    composed flip, not a single fine step.
    """
    return 0.5 * (t_prev - t_cur) / t_prev
