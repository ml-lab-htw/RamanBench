from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from raman_bench.models.custom.base import BaseRamanEstimator


class _MultiScaleBlock(nn.Module):
    """Multi-scale 1D convolution block with cross-scale SE attention."""

    def __init__(self, in_channels, out_channels, reduction, num_branches, stride):
        super().__init__()
        kernel_sizes = [2 * i + 3 for i in range(num_branches)]
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            pad = (ks - 1) // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=ks,
                        stride=stride,
                        padding=pad,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01),
                )
            )
        concat_channels = out_channels * num_branches
        self.pool = nn.AdaptiveAvgPool1d(1)
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
        cat = torch.cat(branch_outs, dim=1)
        mask = self.fc(self.pool(cat).squeeze(-1))
        fused = self.fusion(cat)
        return F.relu(mask.unsqueeze(-1) * fused)


class _SANetNetwork(nn.Module):
    """Multi-scale 1D CNN with SE attention (Deng et al., 2021)."""

    def __init__(
        self,
        n_outputs,
        num_blocks=5,
        channel_factor=2.0,
        initial_channels=16,
        num_branches=6,
        reduction=16,
    ):
        super().__init__()
        channel_seq = [1]
        ch = initial_channels
        for _ in range(num_blocks):
            channel_seq.append(int(ch))
            ch = int(ch * channel_factor)

        blocks = []
        for i in range(num_blocks):
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
        x = x.unsqueeze(1)
        x = self.blocks(x)
        x = self.head(x).squeeze(-1)
        return self.fc(x)


class SANetModel(BaseRamanEstimator):
    """SANet multi-scale SE-attention CNN — sklearn-compatible estimator.

    Supports both classification and regression.

    Reference:
        Deng, L., Zhong, Y., Wang, M., Zheng, X., & Zhang, J. (2021).
        Scale-adaptive deep model for bacterial Raman spectra identification.
        IEEE Journal of Biomedical and Health Informatics, 26(1), 369–378.
        https://doi.org/10.1109/JBHI.2021.3113700
    """

    def __init__(
        self,
        num_blocks=5,
        channel_factor=2.0,
        initial_channels=16,
        num_branches=6,
        reduction=16,
        n_epochs=100,
        lr=1e-3,
        batch_size=128,
        patience=10,
        val_fraction=0.1,
        warmup_epochs=10,
        weight_decay=1e-4,
        aug_noise_sigma=0.01,
        aug_mixup_alpha=1e-12,
        per_epoch_augmentation=False,
        aug_max_train_samples=2000,
        aug_n_per_epoch=1,
        grad_clip_norm=1.0,
    ):
        self.num_blocks = num_blocks
        self.channel_factor = channel_factor
        self.initial_channels = initial_channels
        self.num_branches = num_branches
        self.reduction = reduction
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.aug_noise_sigma = aug_noise_sigma
        self.aug_mixup_alpha = aug_mixup_alpha
        self.per_epoch_augmentation = per_epoch_augmentation
        self.aug_max_train_samples = aug_max_train_samples
        self.aug_n_per_epoch = aug_n_per_epoch
        self.grad_clip_norm = grad_clip_norm

    def fit(self, X, y):
        self.problem_type_ = self._infer_problem_type(y)
        self._setup_device()
        X_np, y_np, n_outputs, criterion = self._prepare_labels(X, y)
        X_train_t, X_val_t, y_train_t, y_val_t = self._train_val_split(
            X_np, y_np, self.val_fraction
        )
        self.model = _SANetNetwork(
            n_outputs=n_outputs,
            num_blocks=self.num_blocks,
            channel_factor=self.channel_factor,
            initial_channels=self.initial_channels,
            num_branches=self.num_branches,
            reduction=self.reduction,
        ).to(self._device)
        self._run_training_loop(
            X_train_t=X_train_t,
            y_train_t=y_train_t,
            X_val_t=X_val_t,
            y_val_t=y_val_t,
            n_epochs=self.n_epochs,
            patience=self.patience,
            time_limit=None,
            criterion=criterion,
            per_epoch_augmentation=self.per_epoch_augmentation,
            batch_size=self.batch_size,
            aug_noise_sigma=self.aug_noise_sigma,
            aug_mixup_alpha=self.aug_mixup_alpha,
            lr=self.lr,
            weight_decay=self.weight_decay,
            warmup_epochs=self.warmup_epochs,
            aug_max_train_samples=self.aug_max_train_samples,
            aug_n_per_epoch=self.aug_n_per_epoch,
            grad_clip_norm=self.grad_clip_norm,
        )
        return self
