# ECE 285 HW1: Generating Chest X-Rays with a VAE

Code for the Homework 1 report of ECE 285 Deep Generative Models (UC San Diego, Fall 2026),
by Florian Saewert (Technical University of Munich). Python 3.12 + PyTorch, managed by
[uv](https://docs.astral.sh/uv/).

## Where to look

| What | File |
|---|---|
| VAE: encoder, decoder, reparameterization, sampling | [`homeworks/hw1/models/vae.py`](homeworks/hw1/models/vae.py) |
| Loss: KL term, reconstruction term, negative ELBO | [`homeworks/hw1/losses.py`](homeworks/hw1/losses.py) |
| All experiments of the report: training, grid search, FID / IS, figures (sections 7-18) | [`homeworks/hw1/notebooks/02_vae_step_by_step.ipynb`](homeworks/hw1/notebooks/02_vae_step_by_step.ipynb) |
| Settings with defaults; illustrated `x-width64` configuration (seed 0) | [`homeworks/hw1/config.py`](homeworks/hw1/config.py), [`homeworks/hw1/configs/final.yaml`](homeworks/hw1/configs/final.yaml) |
| Training loop (TensorBoard + CSV logs, checkpoints) | [`homeworks/hw1/train.py`](homeworks/hw1/train.py) |
| FID and Inception Score, as in the course's Ignite tutorial | [`src/dgm/evaluation/metrics.py`](src/dgm/evaluation/metrics.py), [`homeworks/hw1/evaluate.py`](homeworks/hw1/evaluate.py) |
| Data: download, patient-level split, cached arrays | [`src/dgm/data/chest_xray.py`](src/dgm/data/chest_xray.py) |
| Saved VAE results behind the report (single runs and three-seed means) | [`homeworks/hw1/results/results.csv`](homeworks/hw1/results/results.csv) |
| Corrected real-image baseline, protocol, and sampled filenames | [`homeworks/hw1/results/real_baseline.json`](homeworks/hw1/results/real_baseline.json) |
| Report figure renderer and saved inputs | [`prepare_figures.py`](homeworks/hw1/prepare_figures.py), [`figure_data.npz`](homeworks/hw1/results/figure_data.npz), [`figure_provenance.json`](homeworks/hw1/results/figure_provenance.json) |
| Loss-spike measurements from saved histories | [`training_diagnostics.json`](homeworks/hw1/results/training_diagnostics.json) |

Notebook outputs are stripped. The saved VAE table above can be read immediately on GitHub,
including the GroupNorm comparison; running the notebook regenerates the tables.

## Reproduce

Needs Python 3.12 and uv. A CUDA GPU is used if present; a CPU works too, only slower.

```bash
git clone https://github.com/Florian2142/ece285-dgm-hw1.git
cd ece285-dgm-hw1
uv sync                                             # environment from uv.lock
uv run pytest -m "not slow"                         # tests (synthetic data, ~1 min)
uv run python -m dgm.data.chest_xray --sizes 64     # data: 2.5 GB download, no Kaggle login
```

To reproduce the report's experiments, run
`homeworks/hw1/notebooks/02_vae_step_by_step.ipynb` from the top with `QUICK = False`.
Sections 13-16 evaluate both the standard and fitted priors; sections 17-18 create the
tables and figures. A fresh clone must train the runs because checkpoints are not committed.
The illustrated model is `x-width64`, seed 0: channels 64/128/256, latent dimension 128,
beta 0.03, no augmentation, and no dropout. Its table row averages training seeds 0, 1, 2.

For a single fresh training run with that model configuration, use the command line:

```bash
uv run python -m hw1.train --config homeworks/hw1/configs/final.yaml
uv run python -m hw1.evaluate                       # standard-prior FID / IS of the newest run
```

CLI outputs go to `runs/hw1/<timestamp>-<name>/` (data and runs are not in git).
The CLI and notebook consume random numbers differently, so the same seed does not promise
identical trained weights or scores. The fitted-prior results in the report come from the notebook.

Rebuild the exact image panels and loss plots from the committed figure inputs, without
model training, data downloads, or checkpoints:

```bash
uv run python -m hw1.prepare_figures --output homeworks/hw1/results/figures
```

The input archive stores the decoded images and complete 200-epoch curves used in the PDF.
Its provenance file identifies the runs, selected epoch, validation indices, and sampling seed.
Regenerating those model outputs from scratch still requires the notebook training runs.

## Real-image baseline correction (31 October 2026)

VAE training uses all 4,756 training images, including both classes. The original real-image
comparison accidentally used the first 624 training images, which were all NORMAL because
the files are ordered by class. Its FID-2048 of 70.2 and IS of 1.87 are historical values.

The corrected baseline randomly selects training images without replacement, using seed 0,
to match the 624-image test reference: 234 NORMAL and 390 PNEUMONIA. Labels are used only to
construct this comparison; the VAE remains unconditional. This baseline is a finite-sample
comparison, not a lower bound on attainable FID. Recompute it without training any model:

```bash
uv run python -m hw1.real_baseline --output homeworks/hw1/results/real_baseline.json
```

The corrected scores and selected filenames are saved in the linked JSON above. The VAE
results in `results.csv` are unchanged. Existing notebook score caches retain their historical
contents; the notebook returns the newly computed baseline when reading those caches.

The report's model selection used test-set fitted-prior FID. These remain exploratory
comparisons; correcting the real-image baseline does not turn them into an untouched test.

## Additional evaluation details

The fitted prior is a single Gaussian with mean equal to the mean training encoder mean,
and covariance equal to the sample covariance of those means plus the mean diagonal posterior
variance (with a 1e-6 diagonal stabilizer). Only training encodings are used to fit it.
[Dai and Wipf (2019)](https://arxiv.org/abs/1903.05789) motivate addressing aggregate-posterior
mismatch, but their two-stage VAE is a different method; this notebook uses moment matching.

PCA sampling draws independent zero-mean Gaussian component scores with empirical training
variances, decodes linearly using the training mean and components, and clips to [0, 1].
It adds no residual pixel noise. Validation MSE uses the un-clipped linear projection.

At beta 0.03, changing only BatchNorm to GroupNorm improves standard-prior FID from 228.5 to
172.8 while worsening fitted-prior FID from 101.6 to 128.1 (seed 0). Thus beta is not the sole
explanation for generation quality. The wide model also has transient loss spikes: seed 0
validation KL reaches 4,985 at epoch 27; seed 2 reaches 12,309 at epoch 45. The saved histories
show recovery, but do not isolate a cause; no specific optimizer or normalization explanation
has been experimentally established.

## Use of AI

I built the original encoder and decoder myself and implemented the reparameterization,
forward pass, reconstruction, and basic loss functions, including the KL term and ELBO.
I also chose what to investigate and how to organize the experiments, including the
architecture enhancements, regularization, and hyperparameter comparisons.

I used AI to implement the later enhancements based on those choices. In `vae.py`, this
includes replacing the original fixed layers with flexible layer-building code and adding
the configurable architecture and dropout options. The calibrated Gaussian loss was also
implemented with AI assistance. Much of the supporting code, including data processing,
training and evaluation infrastructure, logging, tests, and figures, was implemented with
AI help. AI also helped restructure the notebook and draft and revise the report.

I am not sure whether I wrote the sampling helper entirely myself, so I have left it out
of the parts I claim to have implemented independently. The Python comments distinguish
my original core implementation from the later AI-assisted additions.

## Layout

```
src/dgm/            shared library: data, training, evaluation (FID / IS), utils
homeworks/hw1/      package `hw1`: model, loss, training, evaluation, configs, notebooks, tests
homeworks/hw1/results/  numbers behind the report
tests/              tests of the shared library
```
