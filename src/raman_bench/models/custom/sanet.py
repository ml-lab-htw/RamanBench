from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from autogluon.common import space

from raman_bench.models.custom.base import BaseCustomModel


class _MultiScaleBlock(nn.Module):
    """Multi-scale 1D convolution block with cross-scale SE attention.

    Creates ``num_branches`` parallel branches with increasing kernel sizes
    (3, 5, 7, …), each consisting of Conv1d + BatchNorm1d (no per-branch
    ReLU).  A cross-scale squeeze-and-excitation path pools the concatenated
    branch outputs, compresses them through two fully-connected layers, and
    produces a channel mask that is applied to a 1×1-fused representation.

    Parameters
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels produced by each branch and by the
        final fused output.
    reduction:
        Reduction ratio for the SE fully-connected bottleneck.
    num_branches:
        Number of parallel convolutional branches.
    stride:
        Stride used by every branch convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int,
        num_branches: int,
        stride: int,
    ) -> None:
        super().__init__()
        kernel_sizes = [2 * i + 3 for i in range(num_branches)]
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            # same-padding: pad = (ks - 1) // 2 ensures consistent output
            # lengths across all branches for any stride.
            pad = (ks - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=ks, stride=stride, padding=pad,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01),
            ))

        concat_channels = out_channels * num_branches
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Cross-scale SE: (num_branches * C) -> C
        self.fc = nn.Sequential(
            nn.Linear(concat_channels, concat_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(concat_channels // reduction, out_channels),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.branches]
        min_len = min(b.size(2) for b in branch_outs)
        branch_outs = [b[:, :, :min_len] for b in branch_outs]
        cat = torch.cat(branch_outs, dim=1)            # (B, n_branches*C, L)
        mask = self.fc(self.pool(cat).squeeze(-1))     # (B, C)
        fused = self.fusion(cat)                       # (B, C, L)
        return F.relu(mask.unsqueeze(-1) * fused)


class _SANetNetwork(nn.Module):
    """Multi-scale 1D CNN with SE attention (Deng et al., 2021).

    Stacks ``num_blocks`` multi-scale blocks with geometrically growing
    channel counts (1 → ``initial_channels`` → ``initial_channels *
    channel_factor`` → …).  A 1×1 convolution reduces the final feature
    maps to 32 channels, followed by batch-normalisation, global average
    pooling, and a fully-connected output layer.

    Parameters
    ----------
    n_outputs:
        Number of output units (classes for classification, 1 for regression).
    num_blocks:
        Number of stacked :class:`_MultiScaleBlock` layers.
    channel_factor:
        Multiplicative growth factor for the channel count between
        successive blocks.
    initial_channels:
        Number of output channels produced by the first block.
    num_branches:
        Number of parallel convolutional branches inside each block.
    reduction:
        Reduction ratio for the SE bottleneck in each block.
    """

    def __init__(
        self,
        n_outputs: int,
        num_blocks: int,
        channel_factor: float,
        initial_channels: int,
        num_branches: int,
        reduction: int,
    ) -> None:
        super().__init__()
        channel_seq: list[int] = [1] + [
            int(initial_channels * (channel_factor ** i)) for i in range(num_blocks)
        ]
        blocks: list[nn.Module] = []
        for i in range(len(channel_seq) - 1):
            blocks.append(
                _MultiScaleBlock(
                    in_channels=channel_seq[i],
                    out_channels=channel_seq[i + 1],
                    stride=2,
                    reduction=reduction,
                    num_branches=num_branches,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv1d(channel_seq[-1], 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.blocks(x)
        x = self.head(x).squeeze(-1)
        return self.fc(x)


class SANetModel(BaseCustomModel):
    """AutoGluon-compatible SANet model for spectral data.

    Multi-scale 1D CNN with squeeze-and-excitation attention.
    Supports both classification and regression.

    Reference
    ---------
    Deng, L., Zhong, Y., Wang, M., Zheng, X., & Zhang, J. (2021).
    Scale-adaptive deep model for bacterial Raman spectra identification.
    *IEEE Journal of Biomedical and Health Informatics*, 26(1), 369–378.
    https://doi.org/10.1109/JBHI.2021.3113700
    """

    def _set_default_params(self) -> None:
        default_params = {
            # Architecture
            "num_blocks": 5,
            "channel_factor": 2.0,
            "initial_channels": 16,
            "num_branches": 6,
            "reduction": 16,
            # Training
            "n_epochs": 100,
            "lr": 1e-3,
            "batch_size": 128,
            "patience": 10,
            "val_fraction": 0.1,
            "warmup_epochs": 10,
            # Regularization
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
            "num_blocks": space.Categorical(4, 6, 8),
            "channel_factor": space.Real(1.2, 2.0),
            "initial_channels": space.Categorical(8, 16, 32),
            "num_branches": space.Categorical(3, 4, 6, 8),
            "reduction": space.Categorical(8, 16, 32),
            # Training
            "lr": space.Real(lower=1e-4, upper=1e-2, log=True),
            # Regularization
            "weight_decay": space.Real(1e-6, 1e-1, log=True),
            # Augmentation
            "aug_noise_sigma": space.Real(1e-3, 0.1, log=True),
            "aug_mixup_alpha": space.Real(1e-3, 1e3, log=True),
            "per_epoch_augmentation": space.Categorical(True, False),
        }

    def _fit(self, X, y, time_limit: float | None = None, **kwargs) -> None:
        params = self._get_model_params()
        self._log_params(params)

        # Architecture
        num_blocks: int = params["num_blocks"]
        channel_factor: float = params["channel_factor"]
        initial_channels: int = params["initial_channels"]
        num_branches: int = params["num_branches"]
        reduction: int = params["reduction"]

        # Training
        n_epochs: int = params["n_epochs"]
        lr: float = params["lr"]
        batch_size: int = params["batch_size"]
        patience: int = params["patience"]
        val_fraction: float = params["val_fraction"]
        warmup_epochs: int = params["warmup_epochs"]

        # Regularization
        weight_decay: float = params["weight_decay"]

        # Augmentation
        aug_noise_sigma: float = params["aug_noise_sigma"]
        aug_mixup_alpha: float = params["aug_mixup_alpha"]
        per_epoch_augmentation: bool = params["per_epoch_augmentation"]
        aug_max_train_samples: int | None = params.get("aug_max_train_samples", None)
        aug_n_per_epoch = params.get("aug_n_per_epoch", 1)
        grad_clip_norm = params.get("grad_clip_norm", None)

        self._setup_device()
        X_np, y_np, n_outputs, criterion = self._prepare_labels(X, y)
        X_train_t, X_val_t, y_train_t, y_val_t = self._train_val_split(X_np, y_np, val_fraction)

        self.model = _SANetNetwork(
            n_outputs=n_outputs,
            num_blocks=num_blocks,
            channel_factor=channel_factor,
            initial_channels=initial_channels,
            num_branches=num_branches,
            reduction=reduction,
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
