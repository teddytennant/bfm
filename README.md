# bfm

Linear-schedule Bernoulli flow models in JAX.

Install from this directory with `pip install -e .`.

Run the tests with `JAX_PLATFORMS=cpu python -m pytest tests -q`.

This is not the paper experiment. It does not train on LSUN, does not report FID, and does not claim FID 9.22. The network is a small MLP, not a U-Net. Only the linear schedule is implemented.
