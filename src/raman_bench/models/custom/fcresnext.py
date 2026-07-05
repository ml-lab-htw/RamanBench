import torch.nn as nn

from raman_bench.models.custom.base import BaseRamanEstimator


class _ResNeXtBlock(nn.Module):
    """ResNeXt-style block: parallel MLPs (cardinality) + residual."""

    def __init__(self, dim, cardinality=4, bottleneck_ratio=4):
        super().__init__()
        bottleneck_dim = max(1, dim // bottleneck_ratio)
        self.branches = nn.ModuleList()
        for _ in range(cardinality):
            self.branches.append(
                nn.Sequential(
                    nn.Linear(dim, bottleneck_dim),
                    nn.BatchNorm1d(bottleneck_dim),
                    nn.ELU(inplace=True),
                    nn.Linear(bottleneck_dim, dim),
                )
            )
        self.bn = nn.BatchNorm1d(dim)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(x + sum(branch(x) for branch in self.branches)))


class _FCResNeXtNetwork(nn.Module):
    """Fully Connected Residual Network with ResNeXt-style parallel MLPs."""

    def __init__(
        self,
        n_features,
        n_outputs,
        hidden_dim=256,
        n_blocks=4,
        cardinality=4,
        pool_size=4,
        fc_dropout=0.2,
    ):
        super().__init__()
        pooled_dim = max(1, n_features // pool_size)
        self.pool = nn.AdaptiveAvgPool1d(pooled_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_ResNeXtBlock(hidden_dim, cardinality=cardinality) for _ in range(n_blocks)]
        )
        self.dropout = nn.Dropout(fc_dropout)
        self.head = nn.Linear(hidden_dim, n_outputs)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(x).squeeze(1)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(self.dropout(x))


class FCResNeXtModel(BaseRamanEstimator):
    """FC-ResNeXt fully connected residual network — sklearn-compatible estimator.

    Supports both classification and regression.

    Reference:
        Zabërgja G, et al. Tabular Data: Is Deep Learning all you need?
        arXiv:2402.03970. https://arxiv.org/abs/2402.03970v1
    """

    def __init__(
        self,
        hidden_dim=256,
        n_blocks=4,
        cardinality=4,
        pool_size=20,
        n_epochs=100,
        lr=1e-3,
        batch_size=128,
        patience=10,
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
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.cardinality = cardinality
        self.pool_size = pool_size
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
        self.model = _FCResNeXtNetwork(
            n_features=X_np.shape[1],
            n_outputs=n_outputs,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            cardinality=self.cardinality,
            pool_size=self.pool_size,
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
