import math

import torch
import torch.nn as nn

from raman_bench.models.custom.base import BaseRamanEstimator


class _RamanTransformerNetwork(nn.Module):
    """ViT-style transformer adapted for 1D Raman spectra (Liu et al., 2023)."""

    def __init__(
        self,
        n_features,
        n_outputs,
        patch_size=16,
        d_model=768,
        nhead=12,
        dim_feedforward=3072,
        n_layers=12,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = math.ceil(n_features / patch_size)
        self.padded_len = self.n_patches * patch_size

        # Align d_model to nearest multiple of nhead
        d_model = int(math.ceil(d_model / nhead) * nhead)

        self.patch_proj = nn.Linear(patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        batch_size = x.size(0)
        if x.size(1) < self.padded_len:
            x = nn.functional.pad(x, (0, self.padded_len - x.size(1)))
        x = x.view(batch_size, self.n_patches, self.patch_size)
        x = self.patch_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class RamanTransformerModel(BaseRamanEstimator):
    """ViT-style Transformer for Raman spectra — sklearn-compatible estimator.

    Supports both classification and regression.

    Reference:
        Liu, B., et al. (2023). Classification of deep-sea cold seep bacteria
        by transformer combined with Raman spectroscopy. Scientific Reports, 13.
        https://doi.org/10.1038/s41598-023-28730-w
    """

    def __init__(
        self,
        patch_size=16,
        d_model=768,
        nhead=12,
        dim_feedforward=3072,
        n_layers=12,
        n_epochs=100,
        lr=1e-3,
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
        self.model = _RamanTransformerNetwork(
            n_features=X_np.shape[1],
            n_outputs=n_outputs,
            patch_size=self.patch_size,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            n_layers=self.n_layers,
            dropout=self.dropout,
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
