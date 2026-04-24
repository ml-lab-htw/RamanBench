import torch.nn as nn
from autogluon.common import space

from raman_bench.models.custom.base import BaseCustomModel


class _ResNeXtBlock(nn.Module):
    """ResNeXt-style block: parallel MLPs (cardinality) + residual."""

    def __init__(self, dim, cardinality=4, bottleneck_ratio=4):
        super().__init__()
        bottleneck_dim = max(1, dim // bottleneck_ratio)
        self.branches = nn.ModuleList()
        for _ in range(cardinality):
            self.branches.append(nn.Sequential(
                nn.Linear(dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ELU(inplace=True),
                nn.Linear(bottleneck_dim, dim),
            ))
        self.bn = nn.BatchNorm1d(dim)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        branch_sum = sum(branch(x) for branch in self.branches)
        return self.activation(self.bn(x + branch_sum))


class _FCResNeXtNetwork(nn.Module):
    """Fully Connected Residual Network (ResNeXt-style MLP).
    The Architecture is suggested in the first version of Zabërgja et. al
    https://arxiv.org/abs/2402.03970v1
    In Lange et al. (2025) it was combined with average pooling to reduce input
    dimension, then ResNeXt-style blocks with parallel MLPs and residual
    connections.
    """

    def __init__(self, n_features, n_outputs, hidden_dim=256,
                 n_blocks=4, cardinality=4, pool_size=4, fc_dropout=0.2):
        super().__init__()
        pooled_dim = max(1, n_features // pool_size)
        self.pool = nn.AdaptiveAvgPool1d(pooled_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
        )

        blocks = []
        for _ in range(n_blocks):
            blocks.append(_ResNeXtBlock(hidden_dim, cardinality=cardinality))
        self.blocks = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(fc_dropout)
        self.head = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x):
        # x: (batch, n_features) -> pool via 1D
        x = x.unsqueeze(1)  # (batch, 1, n_features)
        x = self.pool(x).squeeze(1)  # (batch, pooled_dim)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(self.dropout(x))


class FCResNeXtModel(BaseCustomModel):
    """AutoGluon-compatible FCResNeXt model for spectral data.

    Fully Connected Residual Network with ResNeXt-style parallel MLPs.
    Supports both classification and regression.

    Reference:
        Zabërgja G, Kadra A, Frey CM, Grabocka J.
        Tabular Data: Is Deep Learning all you need?.
        arXiv preprint arXiv:2402.03970. 2024 Feb 6.
        https://arxiv.org/abs/2402.03970v1
    """

    def _set_default_params(self):
        default_params = {
            # Architecture
            "hidden_dim": 256,
            "n_blocks": 4,
            "cardinality": 4,
            "pool_size": 20,
            # Training
            "n_epochs": 100,
            "lr": 1e-3,
            "batch_size": 128,
            "patience": 10,
            "val_fraction": 0.1,
            "warmup_epochs": 10,
            # Regularization
            "fc_dropout": 0.2,
            "weight_decay": 1e-4,
            # Augmentation
            "aug_noise_sigma": 0.01,
            "aug_mixup_alpha": 1e-12,
            "per_epoch_augmentation": False,
            # Training stability
            "grad_clip_norm": 1.0,
            "aug_max_train_samples": 2000,
            "aug_n_per_epoch": 20,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return self._get_base_hp_searchspace()

    @staticmethod
    def _get_base_hp_searchspace():
        return {
            # Architecture
            "hidden_dim": space.Categorical(128, 256, 512),
            "n_blocks": space.Categorical(2, 4, 6),
            # as hidden_dim and cardinality decide upon the dimension of the parallel
            # paths, we optimize them together by fixing their ratio, so we only
            # optimize hidden_dim keep cardinality fixed
            # "cardinality": space.Categorical(2, 4, 8),
            "pool_size": space.Categorical(4, 8, 16, 32),
            # Training
            "lr": space.Real(lower=1e-4, upper=1e-2, log=True),
            # optimizing lr and batch size together is not wise due to their strong
            # interaction, so we keep lr fixed and only optimize batch size
            # "batch_size": space.Categorical(32, 64, 128),
            # Regularization
            "fc_dropout": space.Real(0.0, 0.5),
            "weight_decay": space.Real(1e-6, 1e-2, log=True),
            # Augmentation
            "aug_noise_sigma": space.Real(1e-3, 0.1, log=True),
            "aug_mixup_alpha": space.Real(1e-12, 1e3, log=True),
            "per_epoch_augmentation": space.Categorical(True, False),
        }

    def _fit(self, X, y, time_limit=None, **kwargs):
        params = self._get_model_params()
        self._log_params(params)

        # Architecture
        hidden_dim = params["hidden_dim"]
        n_blocks = params["n_blocks"]
        cardinality = params["cardinality"]
        pool_size = params["pool_size"]

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

        # Training stability
        grad_clip_norm = params.get("grad_clip_norm", None)
        aug_max_train_samples = params.get("aug_max_train_samples", None)
        aug_n_per_epoch = params.get("aug_n_per_epoch", 1)

        self._setup_device()
        X_np, y_np, n_outputs, criterion = self._prepare_labels(X, y)
        X_train_t, X_val_t, y_train_t, y_val_t = self._train_val_split(X_np, y_np, val_fraction)

        self.model = _FCResNeXtNetwork(
            n_features=X_np.shape[1],
            n_outputs=n_outputs,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            cardinality=cardinality,
            pool_size=pool_size,
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
            grad_clip_norm=grad_clip_norm,
            aug_max_train_samples=aug_max_train_samples,
            aug_n_per_epoch=aug_n_per_epoch,
        )
