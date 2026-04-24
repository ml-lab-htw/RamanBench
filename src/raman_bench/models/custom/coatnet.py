import math

import torch
import torch.nn as nn
from autogluon.common import space

from raman_bench.models.custom.base import BaseCustomModel
from raman_bench.models.custom.rezeronet import (  # noqa: F401
    _DepthwiseSeparableConv1d,
    _ReZeroBlock,
)


# ---------------------------------------------------------------------------
# Self-attention module
# ---------------------------------------------------------------------------

class _SelfAttention1d(nn.Module):
    """Multi-head self-attention for 1D sequences with learnable CLS tokens."""

    def __init__(self, d_model, nhead=4, dropout=0.1, num_cls_tokens=1):
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        # Learnable CLS tokens: (1, num_cls_tokens, d_model)
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, d_model))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)
        # Prepend CLS tokens: (B, num_cls_tokens + L, C)
        cls = self.cls_tokens.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = residual + self.dropout(x)
        # Return only CLS token outputs: (B, num_cls_tokens, C)
        return x[:, :self.num_cls_tokens, :]


# ---------------------------------------------------------------------------
# Hybrid network: ReZero encoder + self-attention head
# ---------------------------------------------------------------------------

class _CoAtNetNetwork(nn.Module):
    """ReZero CNN encoder followed by multi-head self-attention.

    Uses the same ReZero block stack as ReZeroNetModel as the encoder,
    then applies self-attention on the resulting feature map before
    global-average-pooling and an MLP classification/regression head.

    Reference:
        Dai Z, Liu H, Le QV, Tan M. Coatnet: Marrying convolution and attention
        for all data sizes. Advances in neural information processing systems.
        2021 Dec 6;34:3965-77.
        https://arxiv.org/abs/2106.04803
    """

    def __init__(
        self,
        n_outputs,
        *,
        input_dim: int | None = None,
        n_blocks: int = 6,
        base_channels: int = 64,
        kernel_size: int = 3,
        channel_factor: float = 1.0,
        nhead: int = 4,
        num_cls_tokens: int = 1,
        attn_dropout: float = 0.1,
        fc_dropout: float = 0.2,
    ):
        super().__init__()

        # -- Stem --
        stem_kernel, stem_stride = 7, 2
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=stem_kernel, stride=stem_stride),
            nn.BatchNorm1d(base_channels),
            nn.ELU(inplace=True),
        )

        # -- ReZero encoder blocks --
        blocks = []
        current_channels = base_channels
        for i in range(n_blocks):
            stride = 2 if i > 0 and i % 2 == 0 else 1
            out_channels = int(current_channels * channel_factor)
            block = _ReZeroBlock(
                current_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
            )
            blocks.append(block)
            current_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        # -- Align channels to be divisible by nhead (required by MultiheadAttention) --
        # Round up to the nearest multiple of nhead via a 1×1 projection.
        aligned_channels = int(math.ceil(current_channels / nhead) * nhead)
        if aligned_channels != current_channels:
            self.proj = nn.Conv1d(current_channels, aligned_channels, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()
        current_channels = aligned_channels

        # -- Self-attention with CLS tokens --
        self.attention = _SelfAttention1d(
            current_channels, nhead=nhead, dropout=attn_dropout,
            num_cls_tokens=num_cls_tokens,
        )

        # CLS tokens aggregate global info — flat_dim is independent of spatial size
        flat_dim = current_channels * num_cls_tokens

        self.head = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(flat_dim, current_channels // 2),
            nn.ELU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(current_channels // 2, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)           # (B, 1, L)
        x = self.stem(x)             # (B, C, L')
        x = self.blocks(x)           # (B, C', L'')
        x = self.proj(x)             # (B, C'', L'') — align to nhead multiple
        cls_out = self.attention(x)  # (B, num_cls_tokens, C'')
        cls_out = cls_out.flatten(1)  # (B, num_cls_tokens * C')
        return self.head(cls_out)    # (B, n_outputs)


# ---------------------------------------------------------------------------
# AutoGluon model wrapper
# ---------------------------------------------------------------------------

class CoAtNetModel(BaseCustomModel):
    """AutoGluon-compatible CoAtNet model for spectral data.

    Combines the ReZero CNN encoder from ReZeroNetModel with multi-head
    self-attention, following the CoAtNet design principle of marrying
    convolution and attention.

    Reference:
        Dai Z, Liu H, Le QV, Tan M. Coatnet: Marrying convolution and attention
        for all data sizes. Advances in neural information processing systems.
        2021 Dec 6;34:3965-77.
        https://arxiv.org/abs/2106.04803
    """

    def _set_default_params(self):
        default_params = {
            # Architecture
            "n_blocks": 6,
            "base_channels": 64,
            "kernel_size": 3,
            "channel_factor": 1.0,
            "nhead": 4,
            "num_cls_tokens": 1,
            # Training
            "n_epochs": 100,
            "lr": 1e-3,
            "batch_size": 128,
            "patience": 15,
            "val_fraction": 0.1,
            "warmup_epochs": 10,
            # Regularization
            "attn_dropout": 0.1,
            "fc_dropout": 0.2,
            "weight_decay": 1e-4,
            # Augmentation
            "aug_noise_sigma": 0.01,
            "aug_mixup_alpha": 1e-12,
            "per_epoch_augmentation": False,
            "aug_max_train_samples": 2000,
            "aug_n_per_epoch": 20,
            "grad_clip_norm": 1.0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return self._get_base_hp_searchspace()

    @staticmethod
    def _get_base_hp_searchspace():
        return {
            # Architecture
            "n_blocks": space.Categorical(4, 6,),
            "channel_factor": space.Real(1.0, 1.6),
            "nhead": space.Categorical(2, 4, 8),
            "num_cls_tokens": space.Categorical(1, 2),
            # Training
            "lr": space.Real(lower=1e-4, upper=1e-2, log=True),
            # Regularization
            "attn_dropout": space.Real(0.0, 0.4),
            "fc_dropout": space.Real(0.0, 0.5),
            "weight_decay": space.Real(1e-6, 1e-1, log=True),
            # Augmentation
            "aug_noise_sigma": space.Real(1e-3, 0.1, log=True),
            "aug_mixup_alpha": space.Real(1e-6, 1e3, log=True),
            "per_epoch_augmentation": space.Categorical(True, False),
        }

    def _fit(self, X, y, time_limit=None, **kwargs):
        params = self._get_model_params()
        self._log_params(params)

        # Architecture
        n_blocks = params["n_blocks"]
        base_channels = params["base_channels"]
        kernel_size = params["kernel_size"]
        channel_factor = params["channel_factor"]
        nhead = params["nhead"]
        num_cls_tokens = params["num_cls_tokens"]

        # Training
        n_epochs = params["n_epochs"]
        lr = params["lr"]
        batch_size = params["batch_size"]
        patience = params["patience"]
        val_fraction = params["val_fraction"]
        warmup_epochs = params["warmup_epochs"]

        # Regularization
        attn_dropout = params["attn_dropout"]
        fc_dropout = params["fc_dropout"]
        weight_decay = params["weight_decay"]

        # Augmentation
        aug_noise_sigma = params["aug_noise_sigma"]
        aug_mixup_alpha = params["aug_mixup_alpha"]
        per_epoch_augmentation = params["per_epoch_augmentation"]
        aug_max_train_samples = params.get("aug_max_train_samples", None)
        aug_n_per_epoch = params.get("aug_n_per_epoch", 1)
        grad_clip_norm = params.get("grad_clip_norm", None)

        self._setup_device()
        X_np, y_np, n_outputs, criterion = self._prepare_labels(X, y)
        X_train_t, X_val_t, y_train_t, y_val_t = self._train_val_split(X_np, y_np, val_fraction)

        self.model = _CoAtNetNetwork(
            n_outputs,
            input_dim=X_np.shape[1],
            n_blocks=n_blocks,
            base_channels=base_channels,
            kernel_size=kernel_size,
            channel_factor=channel_factor,
            nhead=nhead,
            num_cls_tokens=num_cls_tokens,
            attn_dropout=attn_dropout,
            fc_dropout=fc_dropout,
        ).to(self._device)

        self._run_training_loop(
            X_train_t=X_train_t,
            y_train_t=y_train_t,
            X_val_t=X_val_t,
            y_val_t=y_val_t,
            n_epochs=n_epochs,
            patience=patience,
            time_limit=time_limit,
            criterion=criterion,
            per_epoch_augmentation=per_epoch_augmentation,
            batch_size=batch_size,
            aug_noise_sigma=aug_noise_sigma,
            aug_mixup_alpha=aug_mixup_alpha,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            aug_max_train_samples=aug_max_train_samples,
            aug_n_per_epoch=aug_n_per_epoch,
            grad_clip_norm=grad_clip_norm,
        )
