"""Train a small MLP on fixed binary codes and print one JSON metrics line.

Runs on CPU when JAX_PLATFORMS=cpu. Pass a step count as the first argument.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import bfm
import jax
import jax.numpy as jnp

from bfm.path import alpha, time_grid
from bfm.train import batch_loss, init_mlp, sample, train_step


MODES = jnp.array(
    [
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=jnp.float32,
)


def mean_min_hamming(samples, modes):
    dist = jnp.abs(samples[:, None, :] - modes[None, :, :]).sum(axis=-1)
    return dist.min(axis=1).mean()


def main():
    steps = 20
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])

    dim = int(MODES.shape[-1])
    # Fixed codes, repeated, corrupted once at a few times. GD on that batch.
    x0 = jnp.concatenate([MODES for _ in range(4)], axis=0)
    times = jnp.array([0.25, 0.5, 0.75], dtype=jnp.float32)
    chunks_x = []
    chunks_t = []
    key = jax.random.PRNGKey(0)
    for t in times:
        key, k = jax.random.split(key)
        chunks_x.append(jax.random.bernoulli(k, alpha(x0, t)).astype(jnp.float32))
        chunks_t.append(jnp.broadcast_to(t, (x0.shape[0], 1)))
    x_t = jnp.concatenate(chunks_x, axis=0)
    t_col = jnp.concatenate(chunks_t, axis=0)
    x0_rep = jnp.concatenate([x0 for _ in range(len(times))], axis=0)

    key, k_init = jax.random.split(key)
    params = init_mlp(k_init, dim, hidden=64)
    loss_start = float(batch_loss(params, x0_rep, x_t, t_col))
    lr = jnp.float32(0.4)
    for _ in range(steps):
        params, _ = train_step(params, x0_rep, x_t, t_col, lr)
    loss_end = float(batch_loss(params, x0_rep, x_t, t_col))

    key, k_noise, k_samp = jax.random.split(key, 3)
    noise = jax.random.bernoulli(k_noise, 0.5, (64, dim)).astype(jnp.float32)
    grid = time_grid(8)
    samples = sample(params, k_samp, 64, times=grid)
    hamming_noise = float(mean_min_hamming(noise, MODES))
    hamming_samples = float(mean_min_hamming(samples, MODES))

    if not loss_end < loss_start:
        raise SystemExit(f"loss did not decrease: {loss_start} -> {loss_end}")

    line = json.dumps(
        {
            "loss_start": loss_start,
            "loss_end": loss_end,
            "hamming_noise": hamming_noise,
            "hamming_samples": hamming_samples,
        }
    )
    print(line)
    return line


if __name__ == "__main__":
    main()
