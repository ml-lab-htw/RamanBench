import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class BaseRamanEstimator(BaseEstimator):
    """Sklearn-compatible base for all PyTorch-based Raman models.

    Provides helpers for device setup, label encoding, train/val splitting,
    the training loop with early stopping, and prediction.  Concrete
    subclasses implement ``fit(self, X, y)`` by calling these helpers and
    building their network architecture.

    After ``fit`` the fitted attributes ``problem_type_``, ``classes_``
    (classification only), and ``model`` are available.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_problem_type(self, y) -> str:
        y_arr = np.asarray(y)
        if np.issubdtype(y_arr.dtype, np.floating):
            return "regression"
        unique = np.unique(y_arr)
        return "binary" if len(unique) == 2 else "multiclass"

    def _to_numpy_X(self, X) -> np.ndarray:  # noqa: N802
        if hasattr(X, "values"):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def _setup_device(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_labels(self, X, y):
        """Convert X/y to numpy arrays and build the loss criterion.

        Returns
        -------
        X_np : np.ndarray, shape (n, features), dtype float32
        y_np : np.ndarray, shape (n,)
        n_outputs : int
        criterion : nn.Module
        """
        X_np = self._to_numpy_X(X)

        if self.problem_type_ in ("multiclass", "binary"):
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            y_arr = np.asarray(y)
            y_np = np.array([self._class_to_idx[v] for v in y_arr], dtype=np.int64)
            n_outputs = self.n_classes_
            criterion = nn.CrossEntropyLoss()
        else:
            y_np = np.asarray(y, dtype=np.float32)
            n_outputs = 1
            criterion = nn.MSELoss()

        return X_np, y_np, n_outputs, criterion

    def _train_val_split(self, X_np: np.ndarray, y_np: np.ndarray, val_fraction: float):
        """Split arrays into train/val tensors using a fixed random seed."""
        n_val = max(1, int(len(X_np) * val_fraction))
        indices = np.random.RandomState(0).permutation(len(X_np))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train_t = torch.tensor(X_np[train_idx], dtype=torch.float32)
        X_val_t = torch.tensor(X_np[val_idx], dtype=torch.float32)

        if self.problem_type_ in ("multiclass", "binary"):
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

        Uses a cosine annealing schedule with linear warmup.  The best
        model state (lowest validation loss) is restored after training.
        """
        if y_train_t.float().std() == 0:
            raise ValueError(
                f"{self.__class__.__name__}: training target has zero variance "
                f"(all {len(y_train_t)} values are identical) — cannot train."
            )

        if aug_max_train_samples is not None and len(X_train_t) > aug_max_train_samples:
            per_epoch_augmentation = False

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
                    self.__class__.__name__,
                    epoch,
                )
                break

            if per_epoch_augmentation:
                from raman_bench.preprocessing.raman_preprocessing import augment_spectra_torch

                aug_X, aug_y = augment_spectra_torch(
                    X=X_train_t,
                    y=y_train_t,
                    noise_sigma=aug_noise_sigma,
                    shift_max=0,
                    mixup_alpha=aug_mixup_alpha,
                    label_type=self.problem_type_,
                    n_augments=aug_n_per_epoch,
                )
                train_loader = DataLoader(
                    TensorDataset(aug_X, aug_y),
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=(len(aug_X) % batch_size == 1),
                )
            else:
                train_loader = DataLoader(
                    TensorDataset(X_train_t, y_train_t),
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=(len(X_train_t) % batch_size == 1),
                )

            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device)
                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                if torch.isnan(loss):
                    continue
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                optimizer.step()

            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_out = self._batched_forward(X_val_t, device=self._device, batch_size=batch_size)
                val_loss = criterion(val_out.to(self._device), y_val_t.to(self._device)).item()

            if math.isnan(val_loss):
                epochs_no_improve += 1
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info("%s: early stopping at epoch %d", self.__class__.__name__, epoch)
                break

        if best_state is None:
            raise ValueError(
                f"{self.__class__.__name__}: training produced NaN validation loss on every "
                f"epoch — cannot save model (n_train={len(X_train_t)})."
            )
        self.model.load_state_dict(best_state)
        self.model = self.model.cpu()
        # Release CUDA memory immediately so HPO trials don't accumulate VRAM.
        torch.cuda.empty_cache()

        self.model.eval()
        with torch.no_grad():
            check_out = self._batched_forward(
                X_val_t, device=torch.device("cpu"), batch_size=batch_size
            )
        if torch.isnan(check_out).any():
            raise ValueError(
                f"{self.__class__.__name__}: best model produces NaN predictions after training "
                f"(n_train={len(X_train_t)}, n_val={len(X_val_t)})."
            )

    def _batched_forward(
        self, X_t: torch.Tensor, device: torch.device, batch_size: int = 128
    ) -> torch.Tensor:
        """Run model forward pass in batches to avoid OOM on large datasets."""
        chunks = []
        for i in range(0, len(X_t), batch_size):
            chunk = X_t[i : i + batch_size].to(device)
            chunks.append(self.model(chunk).cpu())
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def predict(self, X):
        """Predict labels or regression targets for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self.model.eval()
        X_np = self._to_numpy_X(X)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        device = next(self.model.parameters()).device
        batch_size = getattr(self, "_pred_batch_size", 128)
        with torch.no_grad():
            out = self._batched_forward(X_t, device=device, batch_size=batch_size)
        if self.problem_type_ in ("multiclass", "binary"):
            indices = torch.argmax(out, dim=1).numpy()
            return self.classes_[indices]
        return out.squeeze(1).numpy()

    def predict_proba(self, X):
        """Predict class probabilities for X (classification only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        self.model.eval()
        X_np = self._to_numpy_X(X)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        device = next(self.model.parameters()).device
        batch_size = getattr(self, "_pred_batch_size", 128)
        with torch.no_grad():
            out = self._batched_forward(X_t, device=device, batch_size=batch_size)
        proba = torch.softmax(out, dim=1).numpy()
        if self.problem_type_ == "binary":
            return proba[:, 1]
        return proba
