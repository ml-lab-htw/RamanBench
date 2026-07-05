from __future__ import annotations

import math

import torch
import torch.nn as nn

from raman_bench.models.custom.base import BaseRamanEstimator
from raman_bench.models.custom.rezeronet import (  # noqa: F401
    _DepthwiseSeparableConv1d,
    _ReZeroBlock,
)


class _SelfAttention1d(nn.Module):
    """Multi-head self-attention for 1D sequences with learnable CLS tokens."""

    def __init__(self, d_model, nhead=4, dropout=0.1, num_cls_tokens=1):
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, d_model))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        cls = self.cls_tokens.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = residual + self.dropout(x)
        return x[:, : self.num_cls_tokens, :]


class _CoAtNetNetwork(nn.Module):
    """ReZero CNN encoder followed by multi-head self-attention."""

    def __init__(
        self,
        n_outputs,
        *,
        n_blocks=6,
        base_channels=64,
        kernel_size=3,
        channel_factor=1.0,
        nhead=4,
        num_cls_tokens=1,
        attn_dropout=0.1,
        fc_dropout=0.2,
    ):
        super().__init__()
        stem_kernel, stem_stride = 7, 2
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=stem_kernel, stride=stem_stride),
            nn.BatchNorm1d(base_channels),
            nn.ELU(inplace=True),
        )
        blocks = []
        current_channels = base_channels
        for i in range(n_blocks):
            stride = 2 if i > 0 and i % 2 == 0 else 1
            out_channels = int(current_channels * channel_factor)
            blocks.append(
                _ReZeroBlock(current_channels, out_channels, kernel_size=kernel_size, stride=stride)
            )
            current_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        aligned_channels = int(math.ceil(current_channels / nhead) * nhead)
        self.proj = (
            nn.Conv1d(current_channels, aligned_channels, kernel_size=1, bias=False)
            if aligned_channels != current_channels
            else nn.Identity()
        )
        current_channels = aligned_channels

        self.attention = _SelfAttention1d(
            current_channels,
            nhead=nhead,
            dropout=attn_dropout,
            num_cls_tokens=num_cls_tokens,
        )
        flat_dim = current_channels * num_cls_tokens
        self.head = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(flat_dim, current_channels // 2),
            nn.ELU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(current_channels // 2, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.proj(x)
        cls_out = self.attention(x)
        return self.head(cls_out.flatten(1))


class CoAtNetModel(BaseRamanEstimator):
    """CoAtNet (ReZero CNN + self-attention) — sklearn-compatible estimator.

    Supports both classification and regression.

    Reference:
        Dai Z, et al. CoAtNet: Marrying Convolution and Attention for All Data Sizes.
        NeurIPS 2021. https://arxiv.org/abs/2106.04803
    """

    def __init__(
        self,
        n_blocks=6,
        base_channels=64,
        kernel_size=3,
        channel_factor=1.0,
        nhead=4,
        num_cls_tokens=1,
        attn_dropout=0.1,
        n_epochs=100,
        lr=1e-3,
        batch_size=128,
        patience=15,
        val_fraction=0.1,
        warmup_epochs=10,
        fc_dropout=0.2,
        weight_decay=1e-4,
        aug_noise_sigma=0.01,
        aug_mixup_alpha=1e-12,
        per_epoch_augmentation=False,
        aug_max_train_samples=2000,
        aug_n_per_epoch=1,
        grad_clip_norm=1.0,
    ):
        self.n_blocks = n_blocks
        self.base_channels = base_channels
        self.kernel_size = kernel_size
        self.channel_factor = channel_factor
        self.nhead = nhead
        self.num_cls_tokens = num_cls_tokens
        self.attn_dropout = attn_dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.warmup_epochs = warmup_epochs
        self.fc_dropout = fc_dropout
        self.weight_decay = weight_decay
        self.aug_noise_sigma = aug_noise_sigma
        self.aug_mixup_alpha = aug_mixup_alpha
        self.per_epoch_augmentation = per_epoch_augmentation
        self.aug_max_train_samples = aug_max_train_samples
        self.aug_n_per_epoch = aug_n_per_epoch
        self.grad_clip_norm = grad_clip_norm

    def fit(self, X, y, time_limit=None):
        self.problem_type_ = self._infer_problem_type(y)
        self._setup_device()
        X_np, y_np, n_outputs, criterion = self._prepare_labels(X, y)
        X_train_t, X_val_t, y_train_t, y_val_t = self._train_val_split(
            X_np, y_np, self.val_fraction
        )
        self.model = _CoAtNetNetwork(
            n_outputs=n_outputs,
            n_blocks=self.n_blocks,
            base_channels=self.base_channels,
            kernel_size=self.kernel_size,
            channel_factor=self.channel_factor,
            nhead=self.nhead,
            num_cls_tokens=self.num_cls_tokens,
            attn_dropout=self.attn_dropout,
            fc_dropout=self.fc_dropout,
        ).to(self._device)
        self._run_training_loop(
            X_train_t=X_train_t,
            y_train_t=y_train_t,
            X_val_t=X_val_t,
            y_val_t=y_val_t,
            n_epochs=self.n_epochs,
            patience=self.patience,
            time_limit=time_limit,
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
