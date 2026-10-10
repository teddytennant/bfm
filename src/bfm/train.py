"""x0-prediction training and ancestral sampling on an arbitrary time grid."""

import jax
import jax.numpy as jnp

from bfm.path import alpha, time_grid
from bfm.posterior import posterior_prob


def init_mlp(key, dim: int, hidden: int = 64):
    """Small MLP: (x_t, t) -> x0 probabilities in (0, 1)."""
    k1, k2, k3 = jax.random.split(key, 3)
    fan_in = dim + 1

    def _w(k, n_in, n_out, scale):
        return jax.random.normal(k, (n_in, n_out), dtype=jnp.float32) * jnp.float32(scale)

    return {
        "w1": _w(k1, fan_in, hidden, (2.0 / fan_in) ** 0.5),
        "b1": jnp.zeros((hidden,), dtype=jnp.float32),
        "w2": _w(k2, hidden, hidden, (2.0 / hidden) ** 0.5),
        "b2": jnp.zeros((hidden,), dtype=jnp.float32),
        "w3": _w(k3, hidden, dim, 0.01),
        "b3": jnp.zeros((dim,), dtype=jnp.float32),
    }


def _time_column(x_t, t):
    t = jnp.asarray(t, dtype=x_t.dtype)
    if t.ndim == 0:
        t = jnp.broadcast_to(t, x_t.shape[:1])
    if t.ndim == 1:
        t = t[:, None]
    if t.shape[0] == 1 and x_t.shape[0] != 1:
        t = jnp.broadcast_to(t, (x_t.shape[0], 1))
    return t


def predict_x0(params, x_t, t):
    """Predict clean bits in (0, 1) from noisy bits and time."""
    h = jnp.concatenate([x_t, _time_column(x_t, t)], axis=-1)
    h = jax.nn.relu(h @ params["w1"] + params["b1"])
    h = jax.nn.relu(h @ params["w2"] + params["b2"])
    logits = h @ params["w3"] + params["b3"]
    return jax.nn.sigmoid(logits)


def bce_loss(pred, target):
    """Mean binary cross entropy of predicted x0 against true x0."""
    pred = jnp.clip(pred, 1e-6, 1.0 - 1e-6)
    terms = target * jnp.log(pred) + (1.0 - target) * jnp.log(1.0 - pred)
    return -jnp.mean(terms)


def batch_loss(params, x0, x_t, t):
    return bce_loss(predict_x0(params, x_t, t), x0)


def corrupt(key, x0, t):
    """Sample x_t from the marginal Bernoulli(alpha_t)."""
    probs = alpha(x0, t)
    return jax.random.bernoulli(key, probs).astype(jnp.float32)


@jax.jit
def train_step(params, x0, x_t, t, lr):
    """One SGD step on BCE(predict(x_t, t), x0). Returns updated params and pre-step loss."""

    def _loss(p):
        return batch_loss(p, x0, x_t, t)

    loss, grads = jax.value_and_grad(_loss)(params)
    new_params = jax.tree_util.tree_map(lambda p, gr: p - lr * gr, params, grads)
    return new_params, loss


def _increasing_times(times):
    ts = [float(t) for t in list(times)]
    if len(ts) < 2:
        raise ValueError("times must contain at least two points")
    if ts[0] > ts[-1]:
        ts = list(reversed(ts))
    if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
        raise ValueError("times must be strictly monotone")
    return ts


def sample(params, key, n: int, times=None, num_steps: int = 8):
    """Start at Bernoulli(0.5) and step toward data on the given grid.

    `times` may increase from noise to data or decrease like `time_grid`.
    A decreasing grid is reversed. Each step plugs the MLP x0 prediction
    into the coarse-gap posterior and draws Bernoulli(p).
    """
    if times is None:
        times = time_grid(num_steps)
    ts = _increasing_times(times)
    dim = int(params["w3"].shape[-1])
    key, k0 = jax.random.split(key)
    x = jax.random.bernoulli(k0, jnp.float32(0.5), (n, dim)).astype(jnp.float32)
    for t_noisy, t_clean in zip(ts[:-1], ts[1:]):
        key, k = jax.random.split(key)
        t_noisy_a = jnp.float32(t_noisy)
        t_clean_a = jnp.float32(t_clean)
        x0_hat = predict_x0(params, x, t_noisy_a)
        probs = posterior_prob(x, x0_hat, t_clean_a, t_noisy_a)
        probs = jnp.clip(jnp.nan_to_num(probs, nan=0.5), 0.0, 1.0)
        x = jax.random.bernoulli(k, probs).astype(jnp.float32)
    return x
