import torch.nn as nn

from raman_bench.models.custom.base import BaseRamanEstimator
from raman_bench.preprocessing.bridge_bases import SklearnAutoGluonBridge, _RamanDLBase


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
        flat_dim = 4 * initial_channels * (n_features // 8)
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, dense_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, n_outputs),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.flatten(1)
        return self.classifier(x)


class DeepCNNModel(BaseRamanEstimator):
    """Deep CNN for Raman spectra — sklearn-compatible estimator.

    LeNet-5-inspired architecture with three 1D conv layers (kernels 21, 11, 5).
    Supports both classification and regression.

    Reference:
        Liu, J., Osadchy, M., Ashton, L., Foster, M., Solomon, C. J.,
        & Gibson, S. J. (2017). Deep convolutional neural networks for
        Raman spectrum recognition: a unified solution. Analyst, 142(21),
        4067-4074. https://doi.org/10.1039/C7AN01042G
    """

    def __init__(
        self,
        initial_channels=32,
        dense_dim=256,
        n_epochs=100,
        lr=1e-3,
        batch_size=128,
        patience=10,
        val_fraction=0.1,
        warmup_epochs=10,
        dropout=0.5,
        weight_decay=1e-4,
        aug_noise_sigma=0.01,
        aug_mixup_alpha=1e-12,
        per_epoch_augmentation=False,
        aug_max_train_samples=2000,
        aug_n_per_epoch=1,
        grad_clip_norm=1.0,
    ):
        self.initial_channels = initial_channels
        self.dense_dim = dense_dim
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
        self.model = _DeepCNNNetwork(
            n_features=X_np.shape[1],
            n_outputs=n_outputs,
            dropout=self.dropout,
            initial_channels=self.initial_channels,
            dense_dim=self.dense_dim,
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


class _DeepCNNBridge(SklearnAutoGluonBridge):
    _sklearn_cls = DeepCNNModel
    ag_key = "DEEPCNN"
    ag_name = "DeepCNN"

    def _get_default_searchspace(self):
        from autogluon.common import space

        return {
            "lr": space.Real(1e-4, 3e-3, default=1e-3, log=True),
            "weight_decay": space.Real(1e-6, 1e-3, default=1e-4, log=True),
            "dropout": space.Categorical(0.5, 0.2, 0.3, 0.7),
            "initial_channels": space.Categorical(32, 16, 64),
            "dense_dim": space.Categorical(256, 128, 512),
            "aug_noise_sigma": space.Real(1e-4, 1e-1, default=1e-2, log=True),
            "aug_mixup_alpha": space.Real(1e-2, 1e2, default=1, log=True),
        }


class Prep_DEEPCNN(_RamanDLBase, _DeepCNNBridge):  # noqa: N801
    pass
