"""HW1 settings. YAML files in configs/ override these defaults; the CLI overrides YAML.

Add fields freely (for example to ModelConfig when your architecture needs a new
knob); YAML files and ``key=value`` overrides can then set them. Use a sentinel such
as 0 for "off" instead of ``X | None`` types.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    image_size: int = 64  # images are image_size x image_size, grayscale, values in [0, 1]
    batch_size: int = 128
    num_workers: int = 0  # the data sit in memory; 0 is fastest (and safest on Windows)


@dataclass
class ModelConfig:
    latent_dim: int = 32
    hidden_channels: int = 32  # width of the first encoder layer; how you use it is up to you
    # decoder distribution p(x|z): "bernoulli", "gaussian" (fixed sigma), or
    # "gaussian_calibrated" (sigma^2 = the mean squared error of the batch, Rybkin et al. 2021)
    likelihood: str = "gaussian"
    gaussian_sigma: float = 0.1  # fixed standard deviation, used only by the gaussian likelihood
    # Architecture knobs for the experiments in PLAN.md. The defaults build the hand-written
    # architecture in models/vae.py exactly (same layers, same parameter count).
    num_stages: int = 3  # downsampling stages (3: 64 -> 32 -> 16 -> 8); image_size % 2**n == 0
    convs_per_stage: int = 1  # 3x3 conv layers per stage; 1 = the original architecture
    norm: str = "batch"  # normalization after each conv: "batch", "group", or "none"
    activation: str = "relu"  # "relu", "leaky_relu", or "silu" # lets figure out which one wsorks best here
    downsample: str = "maxpool"  # how a stage halves H and W: "maxpool" or "stride" (4x4 conv)
    upsample: str = "transposed"  # how the decoder doubles H and W: "transposed" or "nearest_conv"
    final_conv: bool = False  # extra 3x3 conv at full resolution before the logits
    residual: bool = False  # x + F(x) around the extra convs of a stage (needs convs_per_stage > 1)
    dropout: float = 0.0  # dropout on the FC bottleneck: encoder features before the heads, decoder fc output


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    lr_schedule: str = "none"  # "none" or "cosine" (anneal the learning rate to 0 over the epochs)
    weight_decay: float = 0.0  # decoupled weight decay (AdamW); 0 = plain Adam
    augment: bool = False  # training-time augmentation: shift <= 4 px, contrast +-10% (hw1.train.augment_batch)
    beta: float = 1.0  # weight of the KL term; 1.0 is the standard ELBO
    kl_warmup_epochs: int = 0  # ramp beta linearly from ~0 to `beta` over this many epochs; 0 = off
    grad_clip_norm: float = 0.0  # 0 = off
    seed: int = 0
    device: str = "auto"  # "auto" (CUDA if available), "cuda", or "cpu"
    sample_every: int = 5  # save a sample grid every N epochs (and after the last epoch)
    max_batches_per_epoch: int = 0  # 0 = all; small values make quick smoke runs


@dataclass
class EvalConfig:
    checkpoint: str = "best"  # "best" (lowest validation negative ELBO) or "last"
    reference_split: str = "test"  # real images that FID compares against
    num_samples: int = 0  # images per FID/IS estimate; 0 = size of the reference split
    repeats: int = 3  # independent sets of generated images -> mean and std
    batch_size: int = 128
    real_baseline: bool = True  # also score real training images: the best reachable value
    seed: int = 1234


@dataclass
class Config:
    name: str = "vae"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
