"""Closed-form linear Bernoulli path against one-bit Bayes, plus a short train."""

import bfm
import jax
import jax.numpy as jnp

from bfm.path import alpha, g, gamma, time_grid
from bfm.posterior import posterior_prob, reverse_bernoulli_prob
from bfm.train import batch_loss, init_mlp, train_step


def test_g_and_alpha_endpoints():
    assert abs(float(g(0.2, 0.3)) - (0.2 * 0.7 + 0.3 * 0.8)) < 1e-6
    for x0 in (0.0, 0.2, 0.5, 0.8, 1.0):
        assert abs(float(alpha(x0, 1.0)) - x0) < 1e-6
        assert abs(float(alpha(x0, 0.0)) - 0.5) < 1e-6
    x0 = jnp.array([0.0, 0.25, 0.7, 1.0], dtype=jnp.float32)
    assert jnp.allclose(alpha(x0, 1.0), x0, atol=1e-6)
    assert jnp.allclose(alpha(x0, 0.0), 0.5, atol=1e-6)


def test_time_grid_and_gamma_formula():
    num_steps = 8
    ts = time_grid(num_steps)
    assert ts.shape == (num_steps + 1,)
    assert abs(float(ts[0]) - 1.0) < 1e-6
    assert abs(float(ts[-1]) - 0.0) < 1e-6
    for tau in range(num_steps + 1):
        expect = 1.0 - tau / num_steps
        assert abs(float(ts[tau]) - expect) < 1e-6

    pairs = [(1.0, 0.5), (0.8, 0.2), (0.75, 0.5), (0.3, 0.1), (1.0, 0.875)]
    for t_prev, t_cur in pairs:
        got = float(gamma(t_prev, t_cur))
        expect = 0.5 * (t_prev - t_cur) / t_prev
        assert abs(got - expect) < 1e-6


def _enumerate_bayes(x_cur, x0, t_prev, t_cur):
    """q(x_prev=1 | x_cur, x0) from Bern(alpha) and the coarse forward kernel.

    Only the two possible cleaner bits are enumerated. The forward kernel is
    P(x_cur=1 | x_prev) = g(x_prev, gamma) with gamma on this same time pair.
    """
    a = float(alpha(x0, t_prev))
    gam = float(gamma(t_prev, t_cur))

    def lik(x_prev):
        p1 = float(g(x_prev, gam))
        if x_cur == 1:
            return p1
        return 1.0 - p1

    num = lik(1.0) * a
    den = num + lik(0.0) * (1.0 - a)
    return num / den


def test_posterior_matches_enumerated_bayes_adjacent_and_skipped():
    num_steps = 10
    dt = 1.0 / num_steps
    grid = [1.0 - i * dt for i in range(num_steps + 1)]
    x0_values = (0.0, 0.2, 0.5, 0.8, 1.0)
    x_cur_values = (0, 1)

    adjacent = [(grid[i], grid[i + 1]) for i in range(num_steps - 1)]
    skipped = [
        (grid[i], grid[j])
        for i in range(num_steps - 1)
        for j in range(i + 2, num_steps)
        if grid[i] > 0.0
    ]
    # A few skipped jumps are enough, including gaps that skip several steps.
    skipped = [skipped[0], skipped[len(skipped) // 3], skipped[len(skipped) // 2], skipped[-1]]

    def check_match(pairs):
        for t_prev, t_cur in pairs:
            for x0 in x0_values:
                for x_cur in x_cur_values:
                    closed = float(posterior_prob(x_cur, x0, t_prev, t_cur))
                    brute = _enumerate_bayes(x_cur, x0, t_prev, t_cur)
                    assert abs(closed - brute) < 1e-5, (
                        t_prev,
                        t_cur,
                        x0,
                        x_cur,
                        closed,
                        brute,
                    )

    check_match(adjacent)
    check_match(skipped)


def test_one_step_gamma_mismatches_skipped_bayes():
    num_steps = 10
    dt = 1.0 / num_steps
    grid = [1.0 - i * dt for i in range(num_steps + 1)]
    skipped = [
        (1.0, 0.5),
        (0.8, 0.2),
        (0.9, 0.3),
        (0.7, 0.2),
        (1.0, 0.2),
    ]
    # Confirm these jumps skip at least one fine step and lie on the grid.
    for t_prev, t_cur in skipped:
        assert t_prev - t_cur > dt + 1e-8
        assert min(abs(t_prev - gdt) for gdt in grid) < 1e-8
        assert min(abs(t_cur - gdt) for gdt in grid) < 1e-8

    max_gap = 0.0
    for t_prev, t_cur in skipped:
        for x0 in (0.2, 0.5, 0.8):
            for x_cur in (0, 1):
                brute = _enumerate_bayes(x_cur, x0, t_prev, t_cur)
                coarse = float(posterior_prob(x_cur, x0, t_prev, t_cur))
                assert abs(coarse - brute) < 1e-5

                gam_one = gamma(t_prev, t_prev - dt)
                alpha_prev = alpha(x0, t_prev)
                one_step = float(reverse_bernoulli_prob(x_cur, alpha_prev, gam_one))
                max_gap = max(max_gap, abs(one_step - brute))
                assert abs(one_step - brute) > 1e-3, (
                    t_prev,
                    t_cur,
                    x0,
                    x_cur,
                    one_step,
                    brute,
                )
    assert max_gap > 1e-2

    # On a true one-step jump the one-step gamma is the coarse gamma.
    t_prev, t_cur = 0.8, 0.8 - dt
    gam_one = gamma(t_prev, t_prev - dt)
    brute = _enumerate_bayes(1, 0.3, t_prev, t_cur)
    one_step = float(reverse_bernoulli_prob(1, alpha(0.3, t_prev), gam_one))
    assert abs(one_step - brute) < 1e-5


def test_training_steps_decrease_loss():
    key = jax.random.PRNGKey(0)
    dim = 4
    n = 16
    key, k_x, k_t, k_init = jax.random.split(key, 4)
    x0 = jax.random.bernoulli(k_x, 0.35, (n, dim)).astype(jnp.float32)
    t = jnp.float32(0.45)
    x_t = jax.random.bernoulli(k_t, alpha(x0, t)).astype(jnp.float32)
    params = init_mlp(k_init, dim, hidden=32)
    loss0 = float(batch_loss(params, x0, x_t, t))
    lr = jnp.float32(0.5)
    for _ in range(8):
        params, _ = train_step(params, x0, x_t, t, lr)
    loss1 = float(batch_loss(params, x0, x_t, t))
    assert jnp.isfinite(loss0) and jnp.isfinite(loss1)
    assert loss1 < loss0


def test_package_exports():
    assert bfm.g(0.0, 0.0) == 0.0
    assert callable(bfm.sample)
    assert callable(bfm.ensure_nix_ld)
