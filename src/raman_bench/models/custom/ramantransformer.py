import math

import torch
import torch.nn as nn
from autogluon.common import space

from raman_bench.models.custom.base import BaseCustomModel


class _RamanTransformerNetwork(nn.Module):
    """ViT-style transformer adapted for 1D Raman spectra (Liu et al., 2023).

    1. Split spectrum into non-overlapping patches
    2. Linear projection to d_model (768) tokens
    3. Prepend learnable [CLS] token + add positional encoding
    4. 12 transformer encoder blocks (12-head attention + MLP)
    5. [CLS] token embedding -> classification/regression head
    """

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

        # Align d_model to the nearest multiple of nhead so that
        # nn.TransformerEncoderLayer never raises the embed_dim divisibility error.
        d_model = int(math.ceil(d_model / nhead) * nhead)

        # Patch embedding
        self.patch_proj = nn.Linear(patch_size, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (learnable, n_patches + 1 for CLS)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
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

        # Classification / regression head
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        # x: (batch, n_features)
        batch_size = x.size(0)

        # Pad to multiple of patch_size
        if x.size(1) < self.padded_len:
            x = nn.functional.pad(x, (0, self.padded_len - x.size(1)))

        # Patchify: (batch, n_patches, patch_size)
        x = x.view(batch_size, self.n_patches, self.patch_size)

        # Patch projection: (batch, n_patches, d_model)
        x = self.patch_proj(x)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches+1, d_model)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Extract [CLS] token embedding
        cls_out = x[:, 0]  # (batch, d_model)

        return self.head(cls_out)


class RamanTransformerModel(BaseCustomModel):
    """AutoGluon-compatible ViT-style Transformer for spectral data.

    Vision Transformer adapted for 1D Raman spectra.
    Patches + [CLS] token + transformer encoder blocks.
    Supports both classification and regression.

    Reference:
        Liu, B., Liu, K., Qi, X., Zhang, W., & Li, B. (2023).
        Classification of deep-sea cold seep bacteria by transformer
        combined with Raman spectroscopy. Scientific Reports, 13(1), 3240.
        https://doi.org/10.1038/s41598-023-28730-w
    """

    def _set_default_params(self):
        default_params = {
            # Architecture
            "patch_size": 16,
            "d_model": 768,
            "nhead": 12,
            "dim_feedforward": 3072,
            "n_layers": 12,
            # Training
            "n_epochs": 100,
            "lr": 1e-3,
            "batch_size": 128,
            "patience": 10,
            "val_fraction": 0.1,
            "warmup_epochs": 10,
            # Regularization
            "dropout": 0.1,
            "weight_decay": 1e-4,
            # Augmentation
            "aug_noise_sigma": 0.01,
            "aug_mixup_alpha": 1e-12,
            # Whether to re-apply augmentation every epoch (True) or
            # apply it once before training and reuse the same augmented
            # dataset for all epochs (False).
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
            "patch_size": space.Categorical(8, 16, 32),
            "d_model": space.Categorical(128, 256, 512),
            "dim_feedforward": space.Categorical(512, 1024, 2048, 3072),
            "n_layers": space.Int(lower=1, upper=12),
            # Training
            "lr": space.Real(lower=1e-5, upper=1e-3, log=True),
            # Regularization
            "dropout": space.Categorical(0.1, 0.2, 0.3),
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
        patch_size = params["patch_size"]
        d_model = params["d_model"]
        nhead = params["nhead"]
        dim_feedforward = params["dim_feedforward"]
        n_layers = params["n_layers"]

        # Training
        n_epochs = params["n_epochs"]
        lr = params["lr"]
        batch_size = params["batch_size"]
        patience = params["patience"]
        val_fraction = params["val_fraction"]
        warmup_epochs = params["warmup_epochs"]

        # Regularization
        dropout = params["dropout"]
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

        self.model = _RamanTransformerNetwork(
            n_features=X_np.shape[1],
            n_outputs=n_outputs,
            patch_size=patch_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
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
