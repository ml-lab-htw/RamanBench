import torch.nn as nn
from autogluon.common import space

from raman_bench.models.custom.base import BaseCustomModel


class _DeepCNNNetwork(nn.Module):
    """LeNet-5-inspired deep CNN for Raman spectra (Liu et al., 2017).

    Three 1D convolutional layers with kernel sizes 21, 11, 5, each followed
    by ReLU and MaxPool1d(2).  After the conv stack: flatten -> Dense(512)
    -> ReLU -> Dropout -> Dense(n_outputs).
    """

    def __init__(self, n_features, n_outputs, dropout=0.5, initial_channels=32, dense_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, initial_channels, kernel_size=21, padding=10),
            nn.BatchNorm1d(initial_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(initial_channels, 2 * initial_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(2 * initial_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(2 * initial_channels, 4 * initial_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(4 * initial_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        # After 3x MaxPool(2) the length is n_features // 8
        flat_dim = 4 * initial_channels * (n_features // 8)
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, dense_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, n_outputs),
        )

    def forward(self, x):
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.flatten(1)
        return self.classifier(x)


class DeepCNNModel(BaseCustomModel):
    """AutoGluon-compatible Deep CNN model for spectral data.

    LeNet-5-inspired architecture with three 1D conv layers
    (kernels 21, 11, 5). Supports both classification and regression.

    Reference:
        Liu, J., Osadchy, M., Ashton, L., Foster, M., Solomon, C. J.,
        & Gibson, S. J. (2017). Deep convolutional neural networks for
        Raman spectrum recognition: a unified solution. Analyst, 142(21),
        4067-4074. https://doi.org/10.1039/C7AN01042G
    """

    def _set_default_params(self):
        default_params = {
            # Architecture
            "initial_channels": 32,
            "dense_dim": 256,
            # Training
            "n_epochs": 100,
            "lr": 1e-3,
            "batch_size": 128,
            "patience": 10,
            "val_fraction": 0.1,
            "warmup_epochs": 10,
            # Regularization
            "dropout": 0.5,
            "weight_decay": 1e-4,
            # Augmentation
            "aug_noise_sigma": 0.01,
            "aug_mixup_alpha": 1e-12,
            "per_epoch_augmentation": False,
            "aug_max_train_samples": 2000,
            "aug_n_per_epoch": 20,
            "grad_clip_norm": 1.0,
        } | getattr(self, "_fixed_params", {})
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_searchspace(self):
        return self._get_base_hp_searchspace()

    @staticmethod
    def _get_base_hp_searchspace():
        return {
            # Architecture
            "initial_channels": space.Categorical(16, 32, 64, 128),
            "dense_dim": space.Categorical(256, 512, 1024, 2048),
            # Training
            "lr": space.Real(lower=1e-4, upper=1e-2, log=True),
            # Regularization
            "dropout": space.Real(0.0, 0.5),
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
        initial_channels = params["initial_channels"]
        dense_dim = params["dense_dim"]

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

        self.model = _DeepCNNNetwork(
            n_features=X_np.shape[1],
            n_outputs=n_outputs,
            dropout=dropout,
            initial_channels=initial_channels,
            dense_dim=dense_dim,
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
