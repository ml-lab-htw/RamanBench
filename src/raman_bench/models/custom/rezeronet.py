import math

import torch
import torch.nn as nn

from raman_bench.models.custom.base import BaseRamanEstimator


class _DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels,
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

        if (in_channels != out_channels) or (stride != 1):
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.projection = nn.Identity()

        self.conv = nn.Sequential(
            _DepthwiseSeparableConv1d(
                in_channels, out_channels, kernel_size, stride=stride, padding=self.padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True),
            _DepthwiseSeparableConv1d(
                out_channels, out_channels, kernel_size, stride=1, padding=self.padding,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.activation = nn.ELU(inplace=True)

    def next_spatial_dim(self, spatial_dim: int) -> int:
        return math.floor((spatial_dim + 2 * self.padding - self.kernel_size) / self.stride + 1)

    def forward(self, x):
        out = self.conv(x)
        res = self.projection(x)
        return self.activation(res + self.alpha * out)


class _ReZeroNetNetwork(nn.Module):
    """CNN with ReZero blocks, increasing channels, and MLP head."""

    def __init__(self, n_outputs, input_dim, n_blocks=8, base_channels=64,
                 kernel_size=3, fc_dropout=0.2, channel_factor=1.0):
        super().__init__()
        stem_kernel, stem_stride = 7, 2
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=stem_kernel, stride=stem_stride),
            nn.BatchNorm1d(base_channels),
            nn.ELU(inplace=True),
        )
        spatial_dim = math.floor((input_dim - stem_kernel) / stem_stride + 1)

        blocks = []
        current_channels = base_channels
        for i in range(n_blocks):
            stride = 2 if i > 0 and i % 2 == 0 else 1
            out_channels = int(current_channels * channel_factor)
            block = _ReZeroBlock(current_channels, out_channels, kernel_size=kernel_size, stride=stride)
            spatial_dim = block.next_spatial_dim(spatial_dim)
            blocks.append(block)
            current_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        flat_dim = current_channels * spatial_dim
        self.fc = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(flat_dim, current_channels // 2),
            nn.ELU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(current_channels // 2, n_outputs),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.flatten(1)
        return self.fc(x)


class ReZeroNetModel(BaseRamanEstimator):
    """ReZeroNet CNN — sklearn-compatible estimator.

    Supports both classification and regression.

    Reference:
        Lange, C., et al. (2025). ReZeroNet: A CNN with ReZero blocks for
        Raman spectroscopy. (RamanBench paper)
    """

    def __init__(
        self,
        n_blocks=6,
        base_channels=64,
        kernel_size=3,
        channel_factor=1.0,
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
        aug_n_per_epoch=20,
        grad_clip_norm=1.0,
    ):
        self.n_blocks = n_blocks
        self.base_channels = base_channels
        self.kernel_size = kernel_size
        self.channel_factor = channel_factor
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

    def fit(self, X, y):
        self.problem_type_ = self._infer_problem_type(y)
        self._setup_device()
        X_np, y_np, n_outputs, criterion = self._prepare_labels(X, y)
        X_train_t, X_val_t, y_train_t, y_val_t = self._train_val_split(
            X_np, y_np, self.val_fraction
        )
        self.model = _ReZeroNetNetwork(
            n_outputs=n_outputs,
            input_dim=X_np.shape[1],
            n_blocks=self.n_blocks,
            base_channels=self.base_channels,
            kernel_size=self.kernel_size,
            fc_dropout=self.fc_dropout,
            channel_factor=self.channel_factor,
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
