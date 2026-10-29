# ECE 285 HW1: Generating Chest X-Rays with a VAE

Code for the Homework 1 report of ECE 285 Deep Generative Models (UC San Diego, Fall 2026),
by Florian Saewert (Technical University of Munich). Python 3.12 + PyTorch, managed by
[uv](https://docs.astral.sh/uv/).

## Where to look

| What | File |
|---|---|
| VAE: encoder, decoder, reparameterization, sampling | [`homeworks/hw1/models/vae.py`](homeworks/hw1/models/vae.py) |
| Loss: KL term, reconstruction term, negative ELBO | [`homeworks/hw1/losses.py`](homeworks/hw1/losses.py) |
| All experiments of the report: training, grid search, FID / IS, figures (sections 7-15) | [`homeworks/hw1/notebooks/02_vae_step_by_step.ipynb`](homeworks/hw1/notebooks/02_vae_step_by_step.ipynb) |
| Settings with defaults; the final run as a config | [`homeworks/hw1/config.py`](homeworks/hw1/config.py), [`homeworks/hw1/configs/final.yaml`](homeworks/hw1/configs/final.yaml) |
| Training loop (TensorBoard + CSV logs, checkpoints) | [`homeworks/hw1/train.py`](homeworks/hw1/train.py) |
| FID and Inception Score, as in the course's Ignite tutorial | [`src/dgm/evaluation/metrics.py`](src/dgm/evaluation/metrics.py), [`homeworks/hw1/evaluate.py`](homeworks/hw1/evaluate.py) |
| Data: download, patient-level split, cached arrays | [`src/dgm/data/chest_xray.py`](src/dgm/data/chest_xray.py) |
| Numbers of the final runs (2 betas x 3 seeds) | [`homeworks/hw1/results/final_table.csv`](homeworks/hw1/results/final_table.csv) |

Notebook outputs are stripped; run the notebook to see them.

## Reproduce

Needs Python 3.12 and uv. A CUDA GPU is used if present; a CPU works too, only slower.

```bash
git clone https://github.com/Florian2142/ece285-dgm-hw1.git
cd ece285-dgm-hw1
uv sync                                             # environment from uv.lock
uv run pytest -m "not slow"                         # tests (synthetic data, ~1 min)
uv run python -m dgm.data.chest_xray --sizes 64     # data: 2.5 GB download, no Kaggle login
```

Then either run `homeworks/hw1/notebooks/02_vae_step_by_step.ipynb` from the top, or use the
command line:

```bash
uv run python -m hw1.train --config homeworks/hw1/configs/final.yaml
uv run python -m hw1.evaluate                       # FID and IS of the newest run
```

Outputs go to `runs/hw1/<timestamp>-<name>/` (data and runs are not in git).

## Use of AI (course policy, Piazza @35)

AI tools helped with data processing and software engineering, which the policy allows. The VAE
itself is my own work.

| AI-assisted (infrastructure) | My own work (the homework) |
|---|---|
| Data download, patient-level split, cached arrays (`src/dgm/data`) | `models/vae.py`: `Encoder`, `Decoder`, `VAE` |
| Training loop, logging, checkpoints (`train.py`, `src/dgm/training`) | `losses.py`: KL term, reconstruction term, `elbo_loss` |
| FID / IS code following the Ignite tutorial (`src/dgm/evaluation`) | Hyperparameters and experiments (`configs/`, notebook) |
| Figure and table helpers, tests | Reading the results and writing the report |

## Layout

```
src/dgm/            shared library: data, training, evaluation (FID / IS), utils
homeworks/hw1/      package `hw1`: model, loss, training, evaluation, configs, notebooks, tests
homeworks/hw1/results/  numbers behind the report
tests/              tests of the shared library
```
