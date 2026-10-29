"""Checks for YOUR VAE (models/vae.py) and loss (losses.py).

Every test is skipped while the code it needs still raises NotImplementedError, so
`uv run pytest homeworks/hw1` shows your progress: skipped = TODO, passed = done.
Check one TODO at a time with -k, for example `uv run pytest homeworks/hw1 -k kl`.
"""

import math

import pytest
import torch
from torch.distributions import Bernoulli, Normal, kl_divergence

from hw1.config import DataConfig, ModelConfig
from hw1.losses import elbo_loss, reconstruction_nll
from hw1.losses import kl_divergence as vae_kl
from hw1.models import VAE, Decoder, Encoder, VAEOutput
from hw1.models.vae import Skip

IMAGE_SIZE = DataConfig().image_size
LATENT = 8
CFG = ModelConfig(latent_dim=LATENT, hidden_channels=16)


def todo(fn, *args, **kwargs):
    """Call ``fn``; skip the test (instead of failing) while it is not implemented."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as err:
        pytest.skip(f"TODO: {err}")


@pytest.fixture
def model():
    torch.manual_seed(0)
    return todo(VAE, CFG, image_size=IMAGE_SIZE)


@pytest.fixture
def images():
    return torch.rand(4, 1, IMAGE_SIZE, IMAGE_SIZE, generator=torch.Generator().manual_seed(0))


# --- losses.py ----------------------------------------------------------------------


def test_kl_is_zero_at_the_prior():
    kl = todo(vae_kl, torch.zeros(3, LATENT), torch.zeros(3, LATENT))
    assert kl.shape == (3,)
    torch.testing.assert_close(kl, torch.zeros(3))


def test_kl_matches_torch_distributions():
    gen = torch.Generator().manual_seed(1)
    mu, logvar = torch.randn(5, LATENT, generator=gen), torch.randn(5, LATENT, generator=gen)
    expected = kl_divergence(Normal(mu, (0.5 * logvar).exp()), Normal(0.0, 1.0)).sum(dim=1)
    torch.testing.assert_close(todo(vae_kl, mu, logvar), expected)


def test_bernoulli_nll_matches_log_likelihood_on_binary_images():
    gen = torch.Generator().manual_seed(2)
    logits = torch.randn(3, 1, 8, 8, generator=gen)
    x = torch.randint(0, 2, (3, 1, 8, 8), generator=gen).float()
    expected = -Bernoulli(logits=logits).log_prob(x).flatten(1).sum(dim=1)
    torch.testing.assert_close(todo(reconstruction_nll, logits, x, "bernoulli"), expected)


def test_gaussian_nll_is_scaled_squared_error_up_to_a_constant():
    gen = torch.Generator().manual_seed(3)
    x = torch.rand(2, 1, 8, 8, generator=gen)
    a, b = torch.randn(2, 1, 8, 8, generator=gen), torch.randn(2, 1, 8, 8, generator=gen)
    sigma = 0.2
    nll_a = todo(reconstruction_nll, a, x, "gaussian", sigma)
    nll_b = reconstruction_nll(b, x, "gaussian", sigma)

    def sq_err(logits):
        return ((x - torch.sigmoid(logits)) ** 2).flatten(1).sum(dim=1)

    torch.testing.assert_close(nll_a - nll_b, (sq_err(a) - sq_err(b)) / (2 * sigma**2))


def test_bernoulli_nll_keeps_gradients_for_confidently_wrong_predictions():
    logits = torch.tensor([[[[100.0, -100.0], [0.0, 0.0]]]], requires_grad=True)
    x = torch.tensor([[[[0.0, 1.0], [0.25, 0.75]]]])
    nll = reconstruction_nll(logits, x, "bernoulli")
    torch.testing.assert_close(nll, torch.tensor([200.0 + 2 * math.log(2)]))
    nll.sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[[1.0, -1.0], [0.25, -0.25]]]]))


def test_reconstruction_nll_rejects_unknown_likelihood():
    with pytest.raises(ValueError, match="Unknown likelihood"):
        reconstruction_nll(torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 4, 4), "unknown")


def test_elbo_loss_combines_the_terms():
    gen = torch.Generator().manual_seed(4)
    x = torch.rand(3, 1, 8, 8, generator=gen)
    out = VAEOutput(
        logits=torch.randn(3, 1, 8, 8, generator=gen),
        mu=torch.randn(3, LATENT, generator=gen),
        logvar=torch.randn(3, LATENT, generator=gen),
        z=torch.zeros(3, LATENT),
    )
    terms = todo(elbo_loss, out, x, beta=0.5, likelihood="gaussian", sigma=0.2)
    assert set(terms) == {"loss", "neg_elbo", "recon", "kl"}
    assert all(value.shape == () for value in terms.values()), "every term is a batch mean"
    recon = reconstruction_nll(out.logits, x, "gaussian", 0.2)
    kl = vae_kl(out.mu, out.logvar)
    torch.testing.assert_close(terms["recon"], recon.mean())
    torch.testing.assert_close(terms["kl"], kl.mean())
    torch.testing.assert_close(terms["neg_elbo"], (recon + kl).mean())
    torch.testing.assert_close(terms["loss"], (recon + 0.5 * kl).mean())


# --- models/vae.py --------------------------------------------------------------------


def test_reparameterize_is_differentiable_and_has_the_right_moments():
    mu = torch.full((20_000, 1), 2.0, requires_grad=True)
    logvar = torch.full((20_000, 1), math.log(0.25), requires_grad=True)
    z = todo(VAE.reparameterize, mu, logvar)
    assert z.shape == mu.shape
    assert abs(z.mean().item() - 2.0) < 0.02 and abs(z.std().item() - 0.5) < 0.02
    z.sum().backward()
    assert mu.grad is not None and logvar.grad is not None and logvar.grad.abs().sum() > 0


def test_huge_logvar_stays_finite():
    # exp(100) overflows float32. The old sqrt(exp(logvar)) path and the KL then become NaN.
    mu = torch.zeros(4, LATENT)
    logvar = torch.full((4, LATENT), 100.0, requires_grad=True)
    kl = todo(vae_kl, mu, logvar)
    z = todo(VAE.reparameterize, mu, logvar.detach())
    assert torch.isfinite(kl).all()
    assert torch.isfinite(z).all()
    assert z.abs().max() < 1e5
    kl.sum().backward()
    assert logvar.grad is not None and torch.isfinite(logvar.grad).all()


def test_encoder_shapes(images):
    encoder = todo(Encoder, CFG, IMAGE_SIZE)
    mu, logvar = todo(encoder, images)
    assert mu.shape == logvar.shape == (len(images), LATENT)


def test_decoder_shapes():
    decoder = todo(Decoder, CFG, IMAGE_SIZE)
    logits = todo(decoder, torch.randn(2, LATENT))
    assert logits.shape == (2, 1, IMAGE_SIZE, IMAGE_SIZE)


def test_forward_shapes_and_sampled_z(model, images):
    out = todo(model, images)
    assert out.logits.shape == images.shape
    assert out.mu.shape == out.logvar.shape == out.z.shape == (len(images), LATENT)
    assert not torch.allclose(out.z, out.mu), "forward should SAMPLE z ~ q(z|x), not use mu"


def test_sample_draws_new_images_in_unit_range(model):
    samples = todo(model.sample, 6)
    assert samples.shape == (6, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert samples.min() >= 0.0 and samples.max() <= 1.0
    assert not torch.allclose(samples[0], samples[1]), "different z should give different images"


def test_reconstruct_is_deterministic_and_in_unit_range(model, images):
    model.eval()
    first = todo(model.reconstruct, images)
    assert first.shape == images.shape and first.min() >= 0.0 and first.max() <= 1.0
    torch.testing.assert_close(model.reconstruct(images), first)


@pytest.mark.parametrize("likelihood", ["gaussian", "bernoulli"])
def test_gradients_reach_every_parameter(model, images, likelihood):
    terms = todo(elbo_loss, todo(model, images), images, likelihood=likelihood)
    assert all(torch.isfinite(value) for value in terms.values())
    terms["loss"].backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_sampling_respects_model_dtype():
    model = VAE(ModelConfig(latent_dim=4, hidden_channels=4), image_size=32).double().eval()
    samples = todo(model.sample, 2)
    assert samples.dtype == torch.float64
    assert samples.shape == (2, 1, 32, 32)
    assert torch.isfinite(samples).all()


# --- experiment knobs (plumbing around the architecture, PLAN.md Phase 0) ---------------


def test_default_settings_build_the_original_architecture():
    """The defaults are the hand-written model: 3 x [conv, BatchNorm, ReLU, MaxPool] in the
    encoder, 2 x [transposed conv, BatchNorm, ReLU] + 1 transposed conv in the decoder, and
    1,052,449 parameters for K = 32, 32 channels, and 64 px images."""
    model = VAE(ModelConfig(latent_dim=32, hidden_channels=32), image_size=64)
    assert sum(p.numel() for p in model.parameters()) == 1_052_449
    nn = torch.nn
    encoder_layers = [type(layer) for layer in model.encoder.features]
    assert encoder_layers == 3 * [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d]
    decoder_layers = [type(layer) for layer in model.decoder.features]
    assert decoder_layers == 2 * [nn.ConvTranspose2d, nn.BatchNorm2d, nn.ReLU] + [
        nn.ConvTranspose2d
    ]


@pytest.mark.parametrize(
    "settings",
    [
        dict(num_stages=4, convs_per_stage=2),
        dict(num_stages=2, convs_per_stage=2, final_conv=True),
        dict(norm="group", activation="silu", downsample="stride"),
        dict(norm="none", activation="leaky_relu", upsample="nearest_conv", final_conv=True),
    ],
)
def test_architecture_settings_keep_the_contract(settings):
    model = VAE(ModelConfig(latent_dim=LATENT, hidden_channels=8, **settings), image_size=32)
    x = torch.rand(2, 1, 32, 32, generator=torch.Generator().manual_seed(5))
    out = model(x)
    assert out.logits.shape == x.shape
    assert out.mu.shape == out.logvar.shape == (2, LATENT)
    assert model.sample(3).shape == (3, 1, 32, 32)


@pytest.mark.parametrize(
    "setting",
    [{"norm": "layer"}, {"activation": "tanh"}, {"downsample": "avg"}, {"upsample": "bilinear"}],
)
def test_unknown_architecture_setting_is_rejected(setting):
    with pytest.raises(ValueError, match="Unknown"):
        VAE(ModelConfig(latent_dim=4, hidden_channels=4, **setting), image_size=32)


def test_calibrated_gaussian_uses_the_batch_mse_as_sigma():
    gen = torch.Generator().manual_seed(6)
    x = torch.rand(3, 1, 8, 8, generator=gen)
    out = VAEOutput(
        logits=torch.randn(3, 1, 8, 8, generator=gen),
        mu=torch.randn(3, LATENT, generator=gen),
        logvar=torch.randn(3, LATENT, generator=gen),
        z=torch.zeros(3, LATENT),
    )
    terms = elbo_loss(out, x, likelihood="gaussian_calibrated", sigma=0.1)  # sigma is ignored
    sigma = ((x - torch.sigmoid(out.logits)) ** 2).mean().sqrt().item()
    expected = reconstruction_nll(out.logits, x, "gaussian", sigma)
    torch.testing.assert_close(terms["recon"], expected.mean())
    torch.testing.assert_close(terms["neg_elbo"], (expected + vae_kl(out.mu, out.logvar)).mean())


# --- residual knob (x + F(x) around the extra convs; notebook section 12) -----------------


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_residual_adds_no_weights_and_keeps_the_contract():
    plain = VAE(ModelConfig(latent_dim=LATENT, hidden_channels=8, convs_per_stage=2), image_size=32)
    residual = VAE(
        ModelConfig(latent_dim=LATENT, hidden_channels=8, convs_per_stage=2, residual=True),
        image_size=32,
    )
    assert count_parameters(residual) == count_parameters(plain)
    assert any(isinstance(layer, Skip) for layer in residual.encoder.features)
    assert any(isinstance(layer, Skip) for layer in residual.decoder.features)
    assert not any(isinstance(layer, Skip) for layer in plain.encoder.features)
    x = torch.rand(2, 1, 32, 32, generator=torch.Generator().manual_seed(7))
    out = residual(x)
    assert out.logits.shape == x.shape
    assert out.mu.shape == out.logvar.shape == (2, LATENT)
    assert residual.sample(3).shape == (3, 1, 32, 32)


def test_residual_wraps_only_the_extra_convs():
    nn = torch.nn
    # one conv per stage: nothing to wrap, residual=True builds exactly the default model
    model = VAE(ModelConfig(latent_dim=32, hidden_channels=32, residual=True), image_size=64)
    assert count_parameters(model) == 1_052_449
    encoder_layers = [type(layer) for layer in model.encoder.features]
    assert encoder_layers == 3 * [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d]
    # two convs per stage: only the second conv of a stage is wrapped, never the first conv,
    # the pooling layer, the transposed conv, or final_conv
    cfg = ModelConfig(
        latent_dim=4, hidden_channels=4, convs_per_stage=2, residual=True, final_conv=True
    )
    model = VAE(cfg, image_size=16)
    encoder_layers = [type(layer) for layer in model.encoder.features]
    assert encoder_layers == 3 * [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, Skip, nn.MaxPool2d]
    decoder_layers = [type(layer) for layer in model.decoder.features]
    assert decoder_layers == 3 * [nn.ConvTranspose2d, nn.BatchNorm2d, nn.ReLU, Skip] + [nn.Conv2d]


def test_skip_returns_the_input_when_the_block_is_zero():
    class Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    x = torch.rand(2, 3, 4, 4, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(Skip(Zero())(x), x)
    block = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
    torch.testing.assert_close(Skip(block)(x), x + block(x))


# --- dropout on the FC bottleneck (notebook section 13) ----------------------------------


def test_dropout_knob_is_off_by_default_and_only_active_in_train_mode():
    plain = VAE(ModelConfig(latent_dim=LATENT, hidden_channels=8), image_size=32)
    dropped = VAE(ModelConfig(latent_dim=LATENT, hidden_channels=8, dropout=0.5), image_size=32)
    assert count_parameters(dropped) == count_parameters(plain)
    z = torch.randn(4, LATENT, generator=torch.Generator().manual_seed(9))
    dropped.train()
    assert not torch.allclose(dropped.decode(z), dropped.decode(z)), "train mode must be random"
    dropped.eval()
    torch.testing.assert_close(dropped.decode(z), dropped.decode(z))
    plain.train()
    torch.testing.assert_close(plain.decode(z), plain.decode(z))  # p = 0: nothing is dropped


@pytest.mark.slow
def test_model_can_fit_a_small_batch(model, images):
    """Sanity check of model + loss together: training on 4 images must lower the loss a lot."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = todo(elbo_loss, todo(model, images), images)["loss"].item()
    for _ in range(300):
        loss = elbo_loss(model(images), images)["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.8 * first
