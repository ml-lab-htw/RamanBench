import torch.nn as nn

from raman_bench.models.custom.base import BaseRamanEstimator


class _RamanNetNetwork(nn.Module):
    """Sliding-window MLP for Raman spectra (Ibtehaz et al., 2023).

    The spectrum is split into overlapping windows of size *window_size* with
    stride ``window_size // 2``.  Each window is independently mapped through
    its own Dense(window_size/2) + BN + LeakyReLU block (no weight sharing).
    All window features are concatenated and passed through dense blocks.
    """

    def __init__(self, n_features, n_outputs, window_size=50, fc_dim=512, fc_dropout=0.5):
        super().__init__()
        self.window_size = window_size
        self.dw = window_size // 2
        self.n_windows = max(1, (n_features - window_size) // self.dw + 1)
        window_out = window_size // 2
        fc_dim2 = fc_dim // 2

        self.feature_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(window_size, window_out),
                    nn.BatchNorm1d(window_out),
                    nn.LeakyReLU(inplace=True),
                )
                for _ in range(self.n_windows)
            ]
        )

        concat_dim = self.n_windows * window_out
        self.head = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(concat_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(fc_dropout * 0.8),
            nn.Linear(fc_dim, fc_dim2),
            nn.BatchNorm1d(fc_dim2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(fc_dropout * 0.5),
            nn.Linear(fc_dim2, n_outputs),
        )

    def forward(self, x):
        import torch
        import torch.nn.functional as F  # noqa: N812

        window_features = []
        for i, extractor in enumerate(self.feature_extractors):
            start = i * self.dw
            end = start + self.window_size
            window = x[:, start:end]
            if window.size(1) < self.window_size:
                window = F.pad(window, (0, self.window_size - window.size(1)))
            window_features.append(extractor(window))
        return self.head(torch.cat(window_features, dim=1))


class RamanNetModel(BaseRamanEstimator):
    """RamanNet sliding-window MLP — sklearn-compatible estimator.

    Supports both classification and regression.

    Reference:
        Ibtehaz, N., Chowdhury, M. E., & Rahman, M. S. (2023). RamanNet:
        A generalized neural network architecture for Raman spectrum analysis.
        arXiv preprint arXiv:2307.07312.
    """

    def __init__(
        self,
        window_size=50,
        fc_dim=512,
        fc_dropout=0.5,
        n_epochs=100,
        lr=1e-3,
        batch_size=64,
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
        self.window_size = window_size
        self.fc_dim = fc_dim
        self.fc_dropout = fc_dropout
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
        self.model = _RamanNetNetwork(
            n_features=X_np.shape[1],
            n_outputs=n_outputs,
            window_size=self.window_size,
            fc_dim=self.fc_dim,
            fc_dropout=self.fc_dropout,
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
