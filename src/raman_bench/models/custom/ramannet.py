import torch
import torch.nn as nn
from autogluon.common import space

from raman_bench.models.custom.base import BaseCustomModel


class _RamanNetNetwork(nn.Module):
    """Sliding-window MLP for Raman spectra (Ibtehaz et al., 2023).

    The spectrum is split into overlapping windows of size *window_size* with
    stride ``window_size // 2``.  Each window is independently mapped through
    its own Dense(window_size/2) + BN + LeakyReLU block (one per window, no
    weight sharing).  All window features are concatenated and passed through
    dense blocks with dropout rates ``fc_dropout``, ``0.8 * fc_dropout``,
    ``0.5 * fc_dropout``.

    The head consists of two dense blocks whose widths are ``fc_dim`` and
    ``fc_dim // 2`` respectively, followed by the output layer.
    """

    def __init__(
        self,
        n_features,
        n_outputs,
        window_size=50,
        fc_dim=512,
        fc_dropout=0.5,
    ):
        super().__init__()
        self.window_size = window_size
        self.dw = window_size // 2

        # Number of windows
        self.n_windows = max(1, (n_features - window_size) // self.dw + 1)
        window_out = window_size // 2

        # Independent feature extractor per window (no weight sharing)
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(window_size, window_out),
                nn.BatchNorm1d(window_out),
                nn.LeakyReLU(inplace=True),
            )
            for _ in range(self.n_windows)
        ])

        concat_dim = self.n_windows * window_out
        fc_dim2 = fc_dim // 2

        # Build head with parameterised dropout (p, 0.8p, 0.5p)
        head_layers = [
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
        ]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        # x: (batch, n_features)
        # Extract overlapping windows
        windows = x.unfold(1, self.window_size, self.dw)  # (batch, n_windows, window_size)
        # Apply each window's own feature extractor
        features = [
            self.feature_extractors[i](windows[:, i, :])
            for i in range(self.n_windows)
        ]
        x = torch.cat(features, dim=1)  # (batch, n_windows * window_size//2)
        return self.head(x)


class RamanNetModel(BaseCustomModel):
    """AutoGluon-compatible RamanNet model for spectral data.

    Sliding-window MLP architecture. Supports both classification and
    regression.

    Reference:
        Ibtehaz, N., Chowdhury, M. E. H., Khandakar, A., Kiranyaz, S.,
        Rahman, M. S., & Zughaier, S. M. (2023). RamanNet: A generalized
        neural network architecture for Raman spectrum analysis. Neural
        Computing and Applications, 35(25), 18719-18735.
        https://doi.org/10.1007/s00521-023-08700-z
    """

    def _set_default_params(self):
        default_params = {
            # Architecture
            "window_size": 50,
            "fc_dim": 512,
            "fc_dropout": 0.5,
            # Training
            "n_epochs": 100,
            "lr": 1e-3,
            "batch_size": 64,
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
            "window_size": space.Categorical(30, 50, 70),
            # Training
            "lr": space.Real(lower=1e-5, upper=1e-3, log=True),
            # Regularization
            "fc_dropout": space.Real(0.1, 0.7),
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
        window_size = params["window_size"]
        fc_dim = params["fc_dim"]
        fc_dropout = params["fc_dropout"]

        # Training
        n_epochs = params["n_epochs"]
        lr = params["lr"]
        batch_size = params["batch_size"]
        patience = params["patience"]
        val_fraction = params["val_fraction"]
        warmup_epochs = params["warmup_epochs"]

        # Regularization
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

        self.model = _RamanNetNetwork(
            n_features=X_np.shape[1],
            n_outputs=n_outputs,
            window_size=window_size,
            fc_dim=fc_dim,
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
            aug_max_train_samples=aug_max_train_samples,
            aug_n_per_epoch=aug_n_per_epoch,
            grad_clip_norm=grad_clip_norm,
        )
