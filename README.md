Linear-schedule Bernoulli flow models in JAX (arXiv:2610.11362).

Install from this directory with `pip install -e .`.

Run the tests with `JAX_PLATFORMS=cpu python -m pytest tests -q`.

This is not the paper experiment. It does not train on LSUN, does not report FID, and does not claim FID 9.22. The network is a small MLP, not a U-Net. Only the linear schedule is implemented. The H200 smoke (job 760876) stayed pending, start estimate 2026-10-10T22:13:00 UTC, and was cancelled before it started.
