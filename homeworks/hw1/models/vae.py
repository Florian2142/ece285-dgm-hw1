"""Your VAE for 1 x H x W chest X-rays.   >>> Implemented homework sections are marked below. <<<

    x [B,1,H,W] --Encoder--> (mu, logvar) [B,K] --reparameterize--> z [B,K] --Decoder--> logits [B,1,H,W]

The probabilistic model behind the code:

- prior                      p(z)       = N(0, I)
- approximate posterior      q_phi(z|x) = N(mu, diag(exp(logvar)))       (Encoder, parameters phi)
- decoder / likelihood       p_theta(x|z), parameterized by the logits   (Decoder, parameters theta)

Check the implemented components in this order
(passed = verified by the check; or use notebooks/02_vae_step_by_step.ipynb):

    1. VAE.reparameterize      uv run pytest homeworks/hw1 -k reparameterize
    2. Encoder                 uv run pytest homeworks/hw1 -k encoder
    3. Decoder                 uv run pytest homeworks/hw1 -k decoder
    4. VAE.forward             uv run pytest homeworks/hw1 -k forward
    5. VAE.sample              uv run pytest homeworks/hw1 -k sample
    6. VAE.reconstruct         uv run pytest homeworks/hw1 -k reconstruct

Contract (train.py, evaluate.py, and the tests use only this interface). Shapes for a batch of
B images and latent dimension K = cfg.latent_dim:

- input ``x``: float tensor ``[B, 1, H, W]``, values in [0, 1], ``H = W = image_size``
- ``Encoder(x)`` -> ``(mu, logvar)``, each ``[B, K]``
- ``Decoder(z)`` -> ``logits [B, 1, H, W]``: unconstrained. ``torch.sigmoid(logits)`` is the
  mean image in [0, 1]; ``losses.py`` turns the logits into ``-log p_theta(x | z)``.
- ``VAE(x)`` -> ``VAEOutput(logits, mu, logvar, z)``
- ``VAE.sample(n)`` -> ``[n, 1, H, W]`` in [0, 1]: NEW images from the prior; no input image
- ``VAE.reconstruct(x)`` -> ``[B, 1, H, W]`` in [0, 1]

I implemented the original encoder and decoder, reparameterization, forward pass, and
reconstruction myself. I chose the experiment ideas and comparisons, including architecture
enhancements, regularization, and hyperparameters. AI implemented the later enhancements
based on those choices. I am not certain about the sampling helper's individual authorship,
so I do not include it in the independently written parts.

The original layer structure is listed beside Encoder and Decoder below. AI implemented
the configurable make_* helpers, flexible layer-building loops, residual option, and dropout
additions. With the default settings, they reproduce my original architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hw1.config import ModelConfig

# exp(logvar) overflows float32 near 88. Past that the KL is inf - inf = NaN, and z is Inf.
# A wide encoder (hidden_channels=64) reaches that in the first epoch: the latent linear
# layer has a large fan-in, so one Adam step swings logvar by tens. Healthy runs stay near 0 to 5.
LOGVAR_MIN = -30.0
LOGVAR_MAX = 10.0


def clamp_logvar(logvar: Tensor) -> Tensor:
    """Keep ``exp(logvar)`` inside float32."""
    return logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)


@dataclass
class VAEOutput:
    logits: Tensor  # decoder output for the sampled z, [B, 1, H, W]
    mu: Tensor  # posterior means, [B, K]
    logvar: Tensor  # posterior log-variances, [B, K]
    z: Tensor  # the sampled latents, [B, K]


# ---------------------------------------------------------------------------------------
# AI-assisted enhancements: helpers for the architecture options I chose to test.
# These and the flexible layer-building loops extend my original encoder and decoder.
# The default settings reproduce the original layers listed below.
# ---------------------------------------------------------------------------------------


def make_norm(kind: str, channels: int) -> nn.Module:
    """Normalization after a conv layer: "batch" (original), "group", or "none"."""
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(math.gcd(8, channels), channels)  # 8 groups, fewer for thin layers
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm: {kind!r}")


def make_activation(kind: str) -> nn.Module:
    """"relu" (original), "leaky_relu", or "silu"."""
    if kind == "relu":
        return nn.ReLU()
    if kind == "leaky_relu":
        return nn.LeakyReLU(0.2)
    if kind == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {kind!r}")


def make_downsample(kind: str, channels: int) -> nn.Module:
    """Halve H and W: "maxpool" (original) or "stride" (a learned 4x4 conv with stride 2)."""
    if kind == "maxpool":
        return nn.MaxPool2d(kernel_size=2)
    if kind == "stride":
        return nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
    raise ValueError(f"Unknown downsample: {kind!r}")


def make_upsample(kind: str, in_channels: int, out_channels: int) -> nn.Module:
    """Double H and W: "transposed" (original: 4x4 transposed conv, stride 2) or
    "nearest_conv" (nearest-neighbour upsampling, then a 3x3 conv; no checkerboard artifacts)."""
    if kind == "transposed":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    if kind == "nearest_conv":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
    raise ValueError(f"Unknown upsample: {kind!r}")


class Skip(nn.Module):
    """``x + F(x)``: a residual connection around a block ``F`` that keeps the shape.

    Used only around the extra same-channel Conv-Norm-Act blocks of a stage when
    ``cfg.residual`` is True (``convs_per_stage > 1``). It adds no weights.
    """

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    """``q_phi(z | x)``: an image -> the parameters ``(mu, logvar)`` of a diagonal Gaussian."""

    def __init__(self, cfg: ModelConfig, image_size: int) -> None:
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.image_size = image_size

        ########################################################################
        # Encoder: convolution, batch normalization, ReLU, max pooling.        #
        #                                                                      #
        # Choices to make (and to justify in the report): convolutional or     #
        # fully connected, depth and width (cfg.hidden_channels), how the      #
        # spatial size shrinks (image_size must be divisible by                #
        # 2**number_of_downsamplings), activations, normalization layers.      #
        #                                                                      #
        # The encoder needs TWO outputs of size K = cfg.latent_dim: mu and     #
        # logvar. Question for you: why predict log(sigma^2) and not sigma^2?  #
        ########################################################################
        '''
        We define a convoluational feature extraction layer with shrinking size
        and maxpool layers aswell as BatchNorm.
        '''
        # My original encoder architecture (the default ModelConfig settings):
        #   64 -> 32: Conv2d(1, c, 3, padding=1),   BatchNorm2d(c),  ReLU, MaxPool2d(2)
        #   32 -> 16: Conv2d(c, 2c, 3, padding=1),  BatchNorm2d(2c), ReLU, MaxPool2d(2)
        #   16 -> 8:  Conv2d(2c, 4c, 3, padding=1), BatchNorm2d(4c), ReLU, MaxPool2d(2)
        # I used AI to help generalize this into the loop below for the experiments I chose:
        # depth, convolutions per stage, normalization, activation, and downsampling.
        # At the defaults, it keeps the original layer order and parameter count.

        if image_size % 2**cfg.num_stages != 0:
            raise ValueError("image size must be divisible by 2**num_stages")

        c = cfg.hidden_channels

        '''
        Making a flexible network for more broad experiments, because first experiments did not beat the linear PCA benchmark
        '''

        layers = []
        in_channels = 1
        for stage in range(cfg.num_stages):  # stage 0: 64 -> 32, stage 1: 32 -> 16, ...
            out_channels = c * 2**stage  # c, 2c, 4c, ...: twice the channels for half the size
            for i in range(cfg.convs_per_stage):
                block = [
                    # padding 1: we go one over the original edge in order to capture the outer edges
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    make_norm(cfg.norm, out_channels),
                    make_activation(cfg.activation),
                ]
                if cfg.residual and i > 0:  # extra same-channel conv: x + F(x), no new weights
                    layers.append(Skip(nn.Sequential(*block)))
                else:
                    layers += block
                in_channels = out_channels
            layers.append(make_downsample(cfg.downsample, out_channels))  # halves H and W
        self.features = nn.Sequential(*layers)

        final_size = image_size // 2**cfg.num_stages  # 8 for 64 px and 3 stages
        num_features = in_channels * final_size * final_size  # 4c * 8 * 8 = 8192 for c = 32

        self.mu_head = nn.Linear(num_features, cfg.latent_dim) # one output tensor for the mu
        self.logvar_head = nn.Linear(num_features, cfg.latent_dim) # one output tensor for the variance
        # My regularization experiment; AI helped implement this dropout option.
        # p = 0 (default) leaves the original model unchanged.
        self.dropout = nn.Dropout(cfg.dropout)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``x [B, 1, H, W]`` -> ``(mu, logvar)``, each ``[B, K]``."""

        ########################################################################
        # Flatten encoder features and predict mu and logvar.                  #
        ########################################################################

        # My original forward path: extract features, flatten, and predict mu/logvar.
        # Dropout and the log-variance safeguard were added later.
        features = self.features(x) # gives an activation map with 8*8*128 features hence = 8192
        flattened = torch.flatten(features, start_dim=1) # [B, 8192]
        flattened = self.dropout(flattened)  # AI-assisted dropout addition; a no-op when p = 0

        mu = self.mu_head(flattened)
        logvar = clamp_logvar(self.logvar_head(flattened))

        return mu, logvar

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################


class Decoder(nn.Module):
    """``p_theta(x | z)``: a latent code -> logits of the image distribution."""

    def __init__(self, cfg: ModelConfig, image_size: int) -> None:
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.image_size = image_size

        ########################################################################
        # Decoder: linear projection and transposed convolutions.              #
        #                                                                      #
        # Usually the mirror image of your encoder: from K numbers back to a   #
        # 1 x H x W image. Do NOT end with a sigmoid: return logits (see the   #
        # contract at the top of this file and losses.py for why).             #
        ########################################################################

        if image_size % 2**cfg.num_stages != 0:
            raise ValueError("image_size must be divisible by 2**num_stages")

        c = cfg.hidden_channels

        self.start_size = image_size // 2**cfg.num_stages  # 8 for 64 px and 3 stages
        self.start_channels = c * 2 ** (cfg.num_stages - 1)  # 4c: the encoder's last width

        num_features = (
            self.start_channels * self.start_size * self.start_size
        )

        # [B, K] → [B, 8192] for your current configuration
        self.fc = nn.Linear(self.latent_dim, num_features)
        # AI-assisted implementation of my decoder-dropout experiment; p = 0 drops nothing.
        self.dropout = nn.Dropout(cfg.dropout)

        # My original decoder (defaults) mirrors the encoder with up-convolutions
        # (transposed convolutions, not interpolation):
        #   8 -> 16:  ConvTranspose2d(4c, 2c, 4, stride=2, padding=1), BatchNorm2d(2c), ReLU
        #   16 -> 32: ConvTranspose2d(2c, c, 4, stride=2, padding=1),  BatchNorm2d(c),  ReLU
        #   32 -> 64: ConvTranspose2d(c, 1, 4, stride=2, padding=1)    -> logits, no activation
        # AI helped implement the flexible loop below; the defaults keep my original layers.
        # final_conv keeps c channels in the last up-convolution and makes the logits
        # with a 3x3 conv at full resolution; convs_per_stage > 1 adds 3x3 convs after each
        # up-convolution; upsample / norm / activation pick the layer types.

        layers = []
        in_channels = self.start_channels
        for stage in reversed(range(cfg.num_stages)):  # stage 2: 8 -> 16, ..., stage 0: 32 -> 64
            is_logits = stage == 0 and not cfg.final_conv  # the original last layer
            if is_logits:
                out_channels = 1
            elif stage > 0:
                out_channels = c * 2 ** (stage - 1)  # 2c, c: the encoder's widths, mirrored
            else:
                out_channels = c  # final_conv: stay at c channels until the 3x3 conv below
            layers.append(make_upsample(cfg.upsample, in_channels, out_channels))  # doubles H, W
            if not is_logits:
                layers += [make_norm(cfg.norm, out_channels), make_activation(cfg.activation)]
                for _ in range(cfg.convs_per_stage - 1):  # extra 3x3 convs at this resolution
                    block = [
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        make_norm(cfg.norm, out_channels),
                        make_activation(cfg.activation),
                    ]
                    if cfg.residual:  # x + F(x) around the extra conv, no new weights
                        layers.append(Skip(nn.Sequential(*block)))
                    else:
                        layers += block
            in_channels = out_channels
        if cfg.final_conv:
            layers.append(nn.Conv2d(in_channels, 1, kernel_size=3, padding=1))  # the logits
        self.features = nn.Sequential(*layers)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, z: Tensor) -> Tensor:
        """``z [B, K]`` -> ``logits [B, 1, H, W]`` (no sigmoid here)."""

        ########################################################################
        # Project and reshape z, then decode to image logits.                  #
        ########################################################################

        # My original forward path: project z, reshape, and decode. Dropout was added later.
        features = self.fc(z)
        features = self.dropout(features)  # AI-assisted dropout addition; a no-op when p = 0

        # we have one z for every pixel
        # hence we must make them spatially aware
        features = features.reshape(
            z.shape[0],
            self.start_channels,
            self.start_size,
            self.start_size,
        )

        logits = self.features(features)

        return logits

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################


class VAE(nn.Module):
    """Encoder + Decoder + the probabilistic parts: reparameterization, prior sampling."""

    def __init__(self, cfg: ModelConfig, image_size: int) -> None:
        super().__init__() # like always we actually initialize the stuff.
        self.latent_dim = cfg.latent_dim
        self.image_size = image_size
        self.encoder = Encoder(cfg, image_size)
        self.decoder = Decoder(cfg, image_size)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``x [B, 1, H, W]`` -> ``(mu, logvar)``: the parameters of ``q_phi(z | x)``."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """``z [B, K]`` -> ``logits [B, 1, H, W]``: the parameters of ``p_theta(x | z)``."""
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Draw ``z ~ N(mu, diag(exp(logvar)))`` so that gradients reach ``mu`` and ``logvar``."""

        ########################################################################
        # Reparameterize: z = mu + sigma * epsilon, shape [B, K].              #
        #                                                                      #
        # Question for you: why can the gradient not flow through             #
        # torch.normal(mu, sigma), and what does the reparameterization trick  #
        # change about that?                                                   #
        ########################################################################

        # My core VAE implementation: reparameterize with z = mu + sigma * epsilon.
        # The log-variance clamp is a later stability safeguard.
        # std = exp(0.5 * logvar). sqrt(exp(logvar)) is the same when finite, and overflows sooner.
        logvar = clamp_logvar(logvar)
        epsilon = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * epsilon

        return z

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x: Tensor) -> VAEOutput:
        """One pass through the whole model: encode, sample ``z ~ q_phi(z | x)``, decode."""

        ########################################################################
        # Encode, reparameterize, and decode; return a                         #
        # VAEOutput(logits=..., mu=..., logvar=..., z=...).                    #
        ########################################################################

        # My core VAE implementation: encode, sample the latent, then decode.
        mu, logvar = self.encode(x) # pass through encoder gives the mu, sigma
        latent_space = self.reparameterize(mu=mu, logvar=logvar) # use the reparametrization trick
        logits = self.decode(latent_space) # give the output

        return VAEOutput(logits=logits, mu=mu, logvar=logvar, z=latent_space) # returns the whole stuff
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str | None = None) -> Tensor:
        """Generate ``n`` NEW images ``[n, 1, H, W]`` in [0, 1]. No input image is involved.

        Use ``next(self.parameters()).device`` when ``device`` is None.
        """

        ########################################################################
        # Sample z from the prior, decode, and apply sigmoid.                   #
        ########################################################################

        # Sampling helper: generate images from standard-normal latents.
        # Individual authorship is not confirmed; not claimed as independently written.
        parameter = next(self.parameters()) # we need to find where the params are running

        if device is None:
            device = parameter.device

        z = torch.randn( # needs to be a tensor and since anyway we sample from multgaussian
            n,
            self.latent_dim,
            device=device,
            dtype=parameter.dtype
        )

        logits = self.decode(z) # let the model run

        return torch.sigmoid(logits)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """Encode ``x`` and return the decoded image ``[B, 1, H, W]`` in [0, 1].

        Deterministic: the same ``x`` must always give the same reconstruction.
        """

        ########################################################################
        # Reconstruct by decoding the posterior mean mu.                       #
        ########################################################################

        # My core VAE implementation: reconstruct through the posterior mean.
        mu, _ = self.encode(x=x) # reconstruction only uses the latent mean
        # now we take same z
        z = mu
        reconstructed_image = self.decode(z)

        return torch.sigmoid(reconstructed_image)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
