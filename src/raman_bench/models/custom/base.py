import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
from autogluon.core.models import AbstractModel
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class BaseCustomModel(AbstractModel):
    """Shared base for all PyTorch-based AutoGluon model wrappers.

    Provides common helpers for device setup, label encoding, train/val
    splitting, the training loop with early stopping, and prediction.
    Concrete subclasses only need to implement ``_set_default_params``,
    ``_get_default_searchspace``, and ``_fit``.
    """

    def set_fixed_params(self, fixed_params: dict) -> None:
        self._fixed_params = fixed_params
        for param, val in fixed_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self):
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {"get_features_kwargs": {"valid_raw_types": ["int", "float"]}}
        )
        return default_auxiliary_params

    def _log_params(self, params: dict) -> None:
        logger.info("%s: training with params: %s", self.__class__.__name__, params)

    def _setup_device(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_labels(self, X, y):
        """Convert X/y to numpy, set label encoding attrs, return tensors + criterion.

        Returns
        -------
        X_np : np.ndarray of shape (n, features), dtype float32
        y_np : np.ndarray of shape (n,), dtype int64 (classification) or float32 (regression)
        n_outputs : int
        criterion : nn.Module
        """
        X_np = X.values.astype(np.float32)

        if self.problem_type in ("multiclass", "binary"):
            self._classes = np.unique(y)
            self._n_classes = len(self._classes)
            self._class_to_idx = {c: i for i, c in enumerate(self._classes)}
            y_np = np.array([self._class_to_idx[v] for v in y], dtype=np.int64)
            n_outputs = self._n_classes
            criterion = nn.CrossEntropyLoss()
        else:
            y_np = y.values.astype(np.float32)
            n_outputs = 1
            criterion = nn.MSELoss()

        return X_np, y_np, n_outputs, criterion

    def _train_val_split(self, X_np: np.ndarray, y_np: np.ndarray, val_fraction: float):
        """Split arrays into train/val tensors using a fixed random seed.

        Returns
        -------
        X_train_t, X_val_t, y_train_t, y_val_t : torch.Tensor
        """
        n_val = max(1, int(len(X_np) * val_fraction))
        indices = np.random.RandomState(0).permutation(len(X_np))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train_t = torch.tensor(X_np[train_idx], dtype=torch.float32)
        X_val_t = torch.tensor(X_np[val_idx], dtype=torch.float32)

        if self.problem_type in ("multiclass", "binary"):
            y_train_t = torch.tensor(y_np[train_idx], dtype=torch.long)
            y_val_t = torch.tensor(y_np[val_idx], dtype=torch.long)
        else:
            y_train_t = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
            y_val_t = torch.tensor(y_np[val_idx], dtype=torch.float32).unsqueeze(1)

        return X_train_t, X_val_t, y_train_t, y_val_t

    def _run_training_loop(
            self,
            X_train_t: torch.Tensor,
            y_train_t: torch.Tensor,
            X_val_t: torch.Tensor,
            y_val_t: torch.Tensor,
            n_epochs: int,
            patience: int,
            time_limit: float | None,
            criterion: nn.Module,
            per_epoch_augmentation: bool,
            batch_size: int,
            aug_noise_sigma: float,
            aug_mixup_alpha: float,
            lr: float,
            weight_decay: float,
            warmup_epochs: int,
            grad_clip_norm: float | None = None,
            aug_max_train_samples: int | None = None,
            aug_n_per_epoch: int = 1,
    ) -> None:
        """Run the full training loop with validation and early stopping.

        Uses a cosine annealing schedule with a linear warmup. If
        ``per_epoch_augmentation`` is enabled, a fresh augmented DataLoader
        (noise + mixup) is built at the start of each epoch. The best model
        state (lowest validation loss) is restored after training.

        Parameters
        ----------
        X_train_t:
            Training features as a float32 tensor of shape (n_train, n_features).
        y_train_t:
            Training labels as a tensor of shape (n_train,) or (n_train, 1).
        X_val_t:
            Validation features as a float32 tensor of shape (n_val, n_features).
        y_val_t:
            Validation labels as a tensor of shape (n_val,) or (n_val, 1).
        n_epochs:
            Maximum number of training epochs.
        patience:
            Number of epochs without validation improvement before early stopping.
        time_limit:
            Optional wall-clock time budget in seconds. Training stops early
            when 90 % of the budget is consumed. Pass ``None`` for no limit.
        criterion:
            Loss function (e.g. ``nn.CrossEntropyLoss`` or ``nn.MSELoss``).
        per_epoch_augmentation:
            If ``True``, apply noise and mixup augmentation at each epoch.
        batch_size:
            Mini-batch size for the training DataLoader.
        aug_noise_sigma:
            Standard deviation of Gaussian noise applied during augmentation.
        aug_mixup_alpha:
            Alpha parameter for the Beta distribution used in mixup augmentation.
        lr:
            Initial learning rate for the Adam optimiser.
        weight_decay:
            L2 regularisation coefficient for the Adam optimiser.
        warmup_epochs:
            Number of epochs over which the learning rate is linearly warmed up
            before cosine annealing begins.
        grad_clip_norm:
            If set, gradient norms are clipped to this value via
            ``torch.nn.utils.clip_grad_norm_`` before each optimiser step.
            Prevents exploding gradients / NaN weights on small datasets.
            ``None`` disables clipping (default).
        aug_max_train_samples:
            If set and the training split has more samples than this threshold,
            ``per_epoch_augmentation`` is forced off.  Augmentation is most
            valuable on small datasets; for large ones it adds overhead without
            much benefit.  ``None`` disables the threshold (default).
        aug_n_per_epoch:
            Number of augmented copies produced per epoch (passed to
            ``augment_spectra_torch`` as ``n_augments``).  Each copy is tiled
            from the training set with fresh noise, so the effective training
            set size per epoch is ``aug_n_per_epoch × n_train``.  Increasing
            this on very small datasets (< batch_size) ensures at least one
            full mini-batch and stabilises BatchNorm statistics.  Default 1.
        """
        if y_train_t.float().std() == 0:
            raise ValueError(
                f"{self.__class__.__name__}: training target has zero variance "
                f"(all {len(y_train_t)} values are identical) — cannot train."
            )

        if aug_max_train_samples is not None and len(X_train_t) > aug_max_train_samples:
            if per_epoch_augmentation:
                logger.info(
                    "%s: disabling per-epoch augmentation — train set (%d) "
                    "exceeds aug_max_train_samples (%d).",
                    self.__class__.__name__, len(X_train_t), aug_max_train_samples,
                )
            per_epoch_augmentation = False

        if per_epoch_augmentation:
            logger.info(
                "%s: per-epoch augmentation ENABLED — noise_sigma=%.4g, "
                "mixup_alpha=%.4g, n_per_epoch=%d, train_samples=%d (→ %d per epoch).",
                self.__class__.__name__, aug_noise_sigma, aug_mixup_alpha,
                aug_n_per_epoch, len(X_train_t), aug_n_per_epoch * len(X_train_t),
            )
        else:
            logger.info(
                "%s: per-epoch augmentation DISABLED — train_samples=%d.",
                self.__class__.__name__, len(X_train_t),
            )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        start_time = time.time()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(max(1, warmup_epochs))
            progress = float(current_epoch - warmup_epochs) / float(
                max(1, n_epochs - warmup_epochs)
            )
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for epoch in range(n_epochs):
            if time_limit is not None and (time.time() - start_time) > time_limit * 0.9:
                logger.info(
                    "%s: time limit approaching, stopping at epoch %d",
                    self.__class__.__name__, epoch,
                )
                break

            if per_epoch_augmentation:
                from raman_bench.preprocessing.raman_preprocessing import augment_spectra_torch
                aug_X, aug_y = augment_spectra_torch(
                    X=X_train_t, y=y_train_t,
                    noise_sigma=aug_noise_sigma, shift_max=0,
                    mixup_alpha=aug_mixup_alpha, label_type=self.problem_type,
                    n_augments=aug_n_per_epoch,
                )
                train_loader = DataLoader(
                    TensorDataset(aug_X, aug_y), batch_size=batch_size, shuffle=True,
                    drop_last=(len(aug_X) % batch_size == 1),
                )
            else:
                train_loader = DataLoader(
                    TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True,
                    drop_last=(len(X_train_t) % batch_size == 1),
                )

            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device)
                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                if torch.isnan(loss):
                    logger.debug(
                        "%s: NaN batch loss at epoch %d — skipping batch.",
                        self.__class__.__name__, epoch,
                    )
                    continue
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_out = self._batched_forward(X_val_t, device=self._device, batch_size=batch_size)
                val_loss = criterion(val_out.to(self._device), y_val_t.to(self._device)).item()

            if math.isnan(val_loss):
                logger.debug(
                    "%s: NaN validation loss at epoch %d — skipping state update.",
                    self.__class__.__name__, epoch,
                )
                epochs_no_improve += 1
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(
                    "%s: early stopping at epoch %d",
                    self.__class__.__name__, epoch,
                )
                break

        if best_state is None:
            raise ValueError(
                f"{self.__class__.__name__}: training produced NaN validation loss on every "
                f"epoch — model cannot be saved.  This is usually caused by numerical "
                f"instability in the network (e.g. exploding gradients, fp16 overflow, or "
                f"a very deep architecture on this dataset; n_train={len(X_train_t)})."
            )
        self.model.load_state_dict(best_state)
        self.model = self.model.cpu()

        # Sanity-check: verify the restored best model does not produce NaN
        # predictions on the validation set.  Internal val loss being non-NaN
        # does not guarantee this — BatchNorm running statistics trained on the
        # augmented distribution can overflow on the original data.
        self.model.eval()
        with torch.no_grad():
            check_out = self._batched_forward(
                X_val_t, device=torch.device("cpu"), batch_size=batch_size)

        if torch.isnan(check_out).any():
            raise ValueError(
                f"{self.__class__.__name__}: best model produces NaN predictions "
                f"after training — BatchNorm running statistics likely diverged "
                f"due to augmented vs. original data mismatch "
                f"(n_train={len(X_train_t)}, n_val={len(X_val_t)})."
            )

    def _batched_forward(
            self, X_t: torch.Tensor, device: torch.device, batch_size: int = 128
    ) -> torch.Tensor:
        """Run model forward pass in batches to avoid OOM on large datasets."""
        chunks = []
        for i in range(0, len(X_t), batch_size):
            chunk = X_t[i: i + batch_size].to(device)
            chunks.append(self.model(chunk).cpu())
        return torch.cat(chunks, dim=0)

    def _predict(self, X, **kwargs):
        self.model.eval()
        X_t = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
        device = next(self.model.parameters()).device
        batch_size = getattr(self, "_pred_batch_size", 128)
        with torch.no_grad():
            out = self._batched_forward(X_t, device=device, batch_size=batch_size)
        if self.problem_type in ("multiclass", "binary"):
            indices = torch.argmax(out, dim=1).numpy()
            return self._classes[indices]
        return out.squeeze(1).numpy()

    def _predict_proba(self, X, **kwargs):
        if self.problem_type == "regression":
            # Inline rather than self._predict() to avoid re-entering
            # RamanPreprocessingMixin._predict and applying preprocessing twice.
            self.model.eval()
            X_t = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
            device = next(self.model.parameters()).device
            batch_size = getattr(self, "_pred_batch_size", 128)
            with torch.no_grad():
                out = self._batched_forward(X_t, device=device, batch_size=batch_size)
            return out.squeeze(1).numpy()
        self.model.eval()
        X_t = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
        device = next(self.model.parameters()).device
        batch_size = getattr(self, "_pred_batch_size", 128)
        with torch.no_grad():
            out = self._batched_forward(X_t, device=device, batch_size=batch_size)
        proba = torch.softmax(out, dim=1).numpy()
        if self.problem_type == "binary":
            return proba[:, 1]
        return proba
