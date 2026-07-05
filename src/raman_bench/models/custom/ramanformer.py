import math

import torch
import torch.nn as nn

from raman_bench.models.custom.base import BaseRamanEstimator


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _RamanFormerNetwork(nn.Module):
    """Transformer encoder for spectral data (Koyun et al., 2024)."""

    def __init__(
        self,
        n_features,
        n_outputs,
        patch_size=128,
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        n_layers=3,
        dropout=0.1,
        post_processing_dim=512,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = math.ceil(n_features / patch_size)
        self.padded_len = self.n_patches * patch_size

        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=self.n_patches)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, post_processing_dim, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(post_processing_dim),
            nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(post_processing_dim, post_processing_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(post_processing_dim // 2, n_outputs),
        )

    def forward(self, x):
        batch_size = x.size(0)
        if x.size(1) < self.padded_len:
            x = nn.functional.pad(x, (0, self.padded_len - x.size(1)))
        x = x.view(batch_size, self.n_patches, self.patch_size)
        x = self.patch_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return self.head(x)


class RamanFormerModel(BaseRamanEstimator):
    """RamanFormer transformer encoder — sklearn-compatible estimator.

    Supports both classification and regression.

    Reference:
        Koyun, O. C., et al. (2024). RamanFormer: A transformer-based
        quantification approach for Raman mixture components. ACS Omega, 9(22).
        https://doi.org/10.1021/acsomega.3c09247
    """

    def __init__(
        self,
        patch_size=128,
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        n_layers=3,
        post_processing_dim=512,
        n_epochs=100,
        lr=1e-4,
        batch_size=128,
        patience=10,
        val_fraction=0.1,
        warmup_epochs=10,
        dropout=0.1,
        weight_decay=1e-4,
        aug_noise_sigma=0.01,
        aug_mixup_alpha=1e-12,
        per_epoch_augmentation=False,
        aug_max_train_samples=2000,
        aug_n_per_epoch=1,
        grad_clip_norm=1.0,
    ):
        self.patch_size = patch_size
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.post_processing_dim = post_processing_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.warmup_epochs = warmup_epochs
        self.dropout = dropout
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
        self.model = _RamanFormerNetwork(
            n_features=X_np.shape[1],
            n_outputs=n_outputs,
            patch_size=self.patch_size,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            n_layers=self.n_layers,
            dropout=self.dropout,
            post_processing_dim=self.post_processing_dim,
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
