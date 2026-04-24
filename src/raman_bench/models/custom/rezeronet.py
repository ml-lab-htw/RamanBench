import math

import torch
import torch.nn as nn
from autogluon.common import space

from raman_bench.models.custom.base import BaseCustomModel


class _DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class _ReZeroBlock(nn.Module):
    """Residual block with ReZero scaling."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        # If channels change or stride != 1, we need to adapt the identity path
        if (in_channels != out_channels) or (stride != 1):
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.projection = nn.Identity()

        self.conv = nn.Sequential(
            _DepthwiseSeparableConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=self.padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True),
            _DepthwiseSeparableConv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=self.padding,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.activation = nn.ELU(inplace=True)

    def next_spatial_dim(self, spatial_dim: int) -> int:
        """Compute output spatial dimension given input spatial dimension."""
        return math.floor((spatial_dim + 2 * self.padding - self.kernel_size) / self.stride + 1)

    def forward(self, x):
        out = self.conv(x)
        res = self.projection(x)
        out = res + self.alpha * out
        out = self.activation(out)
        return out


class _ReZeroNetNetwork(nn.Module):
    """
    CNN with ReZero blocks, increasing channels, and MLP head.
    When input_dim is provided, spatial dimensions are tracked through
    all layers so global pooling can be replaced by a direct FC layer.
    """

    def __init__(
        self,
        n_outputs,
        input_dim,
        n_blocks=8,
        base_channels=64,
        kernel_size=3,
        fc_dropout=0.2,
        channel_factor=1.0,
    ):
        super().__init__()

        # Stem: Conv1d(1, base_channels, kernel_size=7, stride=2) — no padding
        stem_kernel, stem_stride = 7, 2
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=stem_kernel, stride=stem_stride),
            nn.BatchNorm1d(base_channels),
            nn.ELU(inplace=True),
        )
        # Spatial dim after stem (padding=0)
        spatial_dim = math.floor((input_dim - stem_kernel) / stem_stride + 1)

        blocks = []
        current_channels = base_channels

        for i in range(n_blocks):
            stride = 2 if i > 0 and i % 2 == 0 else 1
            out_channels = int(current_channels * channel_factor)

            block = _ReZeroBlock(
                current_channels, out_channels, kernel_size=kernel_size, stride=stride
            )
            spatial_dim = block.next_spatial_dim(spatial_dim)
            blocks.append(block)
            current_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        # No global pooling — we know the exact spatial dim
        flat_dim = current_channels * spatial_dim

        self.fc = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(flat_dim, current_channels // 2),
            nn.ELU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(current_channels // 2, n_outputs),
        )
        self._flat_dim = flat_dim

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.flatten(1)  # (batch, channels * spatial_dim)
        return self.fc(x)


class ReZeroNetModel(BaseCustomModel):
    def _set_default_params(self):
        default_params = {
            # Architecture
            "n_blocks": 6,
            "base_channels": 64,
            "kernel_size": 3,
            "channel_factor": 1.0,
            # Training
            "n_epochs": 100,
            "lr": 1e-3,
            "batch_size": 128,
            "patience": 15,
            "val_fraction": 0.1,
            "warmup_epochs": 10,
            # Regularization
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
            "n_blocks": space.Categorical(6, 8, 10),
            "channel_factor": space.Real(1.0, 1.6, log=True),
            # Training
            "lr": space.Real(lower=1e-4, upper=1e-2, log=True),
            # optimizing lr and batch size together is not wise due to their strong interaction,
            # so we keep lr fixed and only optimize batch size
            # "batch_size": space.Categorical(32, 64, 128),
            # Regularization
            "fc_dropout": space.Real(0.0, 0.5),
            "weight_decay": space.Real(1e-6, 1e-1, log=True),
            # Augmentation
            "aug_noise_sigma": space.Real(1e-3, 0.1, log=True),
            "aug_mixup_alpha": space.Real(1e-3, 1e3, log=True),
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

        # Training
        n_epochs = params["n_epochs"]
        lr = params["lr"]
        batch_size = params["batch_size"]
        patience = params["patience"]
        val_fraction = params["val_fraction"]
        warmup_epochs = params["warmup_epochs"]

        # Regularization
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

        self.model = _ReZeroNetNetwork(
            n_outputs,
            input_dim=X_np.shape[1],
            n_blocks=n_blocks,
            base_channels=base_channels,
            kernel_size=kernel_size,
            fc_dropout=fc_dropout,
            channel_factor=channel_factor,
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
