"""The VAE objective: the negative evidence lower bound (ELBO).   >>> Implemented homework sections. <<<

For one image x, with q(z|x) = N(mu, diag(sigma^2)) and the prior p(z) = N(0, I):

    ELBO(x) = E_{q(z|x)}[ log p(x|z) ]  -  KL( q(z|x) || p(z) )      (maximize)
    loss(x) = recon(x)  +  beta * kl(x)                             (minimize)

- recon(x) = -log p(x|z) for ONE sample z ~ q(z|x): a one-sample Monte Carlo estimate
  of the expectation (VAE.reparameterize in models/vae.py makes it differentiable)
- kl(x)    = KL( q(z|x) || p(z) ), in closed form (both are Gaussian)

Conventions that train.py, the plots, and the report rely on (state them in the report):
both terms are SUMMED over pixels and latent dimensions (values per image, in nats)
and then AVERAGED over the images in the batch.

Check the implemented loss functions in this order:

    1. kl_divergence         uv run pytest homeworks/hw1 -k kl
    2. reconstruction_nll    uv run pytest homeworks/hw1 -k nll
    3. elbo_loss             uv run pytest homeworks/hw1 -k elbo
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from hw1.models.vae import VAEOutput, clamp_logvar


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """``KL( N(mu, diag(exp(logvar))) || N(0, I) )`` for each example; shape ``[B]``."""

    ########################################################################
    # Closed-form Gaussian KL, summed over K latent dimensions.            #
    #                                                                      #
    # Derive it on paper first, for ONE dimension:                         #
    # KL( N(m, s^2) || N(0, 1) ) = ?   Then write it with logvar = log s^2. #
    ########################################################################
    #   -1/2 * SUM(1 + log(s^2) - mu^2 - s^2)
    # Clamp before exp: a raw logvar above ~88 makes this inf - inf = NaN (see clamp_logvar).
    logvar = clamp_logvar(logvar)
    KL = -1/2 * (1 + logvar - mu**2  - logvar.exp()).sum(dim=1)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return KL

def reconstruction_nll(
    logits: Tensor, x: Tensor, likelihood: str = "bernoulli", sigma: float = 0.1
) -> Tensor:
    """``-log p(x | z)`` for each example, summed over pixels; shape ``[B]``.

    - ``"bernoulli"``: every pixel is Bernoulli with probability ``sigmoid(logits)``.
      Grayscale x in [0, 1] is not truly binary; say how you treat that in the report.
    - ``"gaussian"``: every pixel is ``N(sigmoid(logits), sigma^2)`` with a fixed sigma.
      Decide whether to keep the constant term and say so.
    """

    ########################################################################
    # Gaussian or Bernoulli NLL; reject unknown likelihoods.               #
    #                                                                      #
    # Hint: prefer functions that take logits directly (numerically       #
    # stable) over torch.sigmoid followed by torch.log.                    #
    ########################################################################

    if likelihood == "gaussian":
        squared_erors = (x - torch.sigmoid(logits))**2
        summed_errors = squared_erors.sum(dim=(1, 2, 3))

        return 1 / (2 * sigma**2) * summed_errors + (x.shape[1] * x.shape[2] * x.shape[3]) * math.log(sigma * math.sqrt(2*math.pi))
    elif likelihood == "bernoulli":
        # Compute log(p) and log(1-p) directly from logits. This keeps the
        # manual BCE formula without log(0) or clipping away gradients.
        zeros = torch.zeros_like(logits)
        log_p = -torch.logaddexp(zeros, -logits)
        log_one_minus_p = -torch.logaddexp(zeros, logits)

        pixel_losses = -(x * log_p + (1 - x) * log_one_minus_p)
        summed_losses = pixel_losses.sum(dim=(1,2,3))
        return summed_losses
    else:
        raise ValueError(f"Unknown likelihood: {likelihood!r}")
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################


def elbo_loss(
    out: VAEOutput,
    x: Tensor,
    *,
    beta: float = 1.0,
    likelihood: str = "bernoulli",
    sigma: float = 0.1,
) -> dict[str, Tensor]:
    """Batch means of the loss terms, as a dict with exactly these keys (train.py logs them):

    - ``"loss"``:     what the optimizer minimizes, ``recon + beta * kl``
    - ``"neg_elbo"``: the true negative ELBO (beta = 1), used to pick the best checkpoint
    - ``"recon"``:    the reconstruction term
    - ``"kl"``:       the KL term
    """

    ########################################################################
    # Combine reconstruction_nll and kl_divergence (each [B]), then        #
    # average over the batch.                                              #
    #                                                                      #
    # Question for you: how does beta != 1 change the reconstruction/KL   #
    # tradeoff, while "neg_elbo" always keeps the original KL weight?      #
    ########################################################################

    # Plumbing (26 Oct 2026, PLAN.md Phase 0): the calibrated Gaussian decoder of Rybkin et
    # al. (2021). sigma is the maximum-likelihood value for this batch, sigma^2 = mean squared
    # error (a constant for the gradient); then it is the ordinary Gaussian NLL from above.
    if likelihood == "gaussian_calibrated":
        sigma = (x - torch.sigmoid(out.logits)).square().mean().sqrt().item()
        likelihood = "gaussian"

    reconst_loss = reconstruction_nll(out.logits, x, likelihood=likelihood, sigma=sigma)
    dl_divergence = kl_divergence(out.mu, out.logvar)

    neg_elbo = (1/dl_divergence.shape[0]) * (reconst_loss + dl_divergence).sum() # make the elbo

    loss = reconst_loss.mean() + beta * dl_divergence.mean() ## average over all samples

    dict_output = {"loss": loss, "neg_elbo": neg_elbo, "recon": reconst_loss.mean(), "kl": dl_divergence.mean()}

    return dict_output
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
