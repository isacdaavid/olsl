# olsl

Python port of [Online Latent State Learning](https://github.com/cochran4/OnlineLatentStateLearning).

**olsl** implements a biologically plausible approximate Bayesian
model of reinforcement learning with contextual and latent changes in
the environment — that is, when the mapping from cues to rewards can
change abruptly, and sometimes is not even directly observable. The
agent maintains beliefs over a set of discrete *latent states*
(competing cue–reward association vectors) and updates those beliefs
online using an approximate Bayesian filter. It reproduces a wide
range of conditioning phenomena that classical Rescorla–Wagner fails
to explain, including the partial reinforcement extinction effect,
rapid renewal of fear after extinction, and backward blocking.

> Cochran, A. L., & Cisler, J. M. (2019). **A flexible and generalizable model of online latent-state learning.** *PLOS Computational Biology*, 15(9), e1007331. https://doi.org/10.1371/journal.pcbi.1007331

## Installation

Requires Python ≥ 3.11

```{bash}
git clone https://github.com/<your-fork>/olsl.git
cd olsl
```

For a minimal install without dev tools:

```{bash}
pip install -e.
```

## Quick start

```{bash}
python examples/run_pree.py
python examples/run_...
```

## Testing

```{bash}
pytest                                                  # all tests
pytest tests/test_latent_state.py -v                    # one file
pytest tests/test_latent_state.py::test_rw_single_trial # one test
pytest -k golden                                        # only golden-trajectory tests
```

