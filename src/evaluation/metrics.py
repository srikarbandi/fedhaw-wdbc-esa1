"""Classification metrics. Positive class = Malignant (label 1, output index 1).

TP = malignant predicted malignant      FN = malignant predicted benign
TN = benign predicted benign            FP = benign predicted malignant
Recall = TP/(TP+FN)   Specificity = TN/(TN+FP)   FPR = FP/(FP+TN)
Precision = TP/(TP+FP)   F1 = 2PR/(P+R)
ROC-AUC uses P(malignant) = softmax(logits)[:, 1].
A metric whose denominator is zero, or ROC-AUC on a single-class subset, is
undefined and returned as NaN (written as null in JSON).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.federated.params import set_flat_params

NAN = float("nan")
METRICS = ["accuracy", "recall", "specificity", "fpr", "precision", "f1", "roc_auc"]


def classification_metrics(y_true, prob_malignant, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob_malignant, dtype=np.float64)
    n = len(y_true)
    out = {"n": int(n), "n_malignant": int((y_true == 1).sum()), "n_benign": int((y_true == 0).sum())}
    if n == 0:
        return {**out, **{m: NAN for m in METRICS}, "tn": 0, "fp": 0, "fn": 0, "tp": 0,
                "finite_outputs": True}
    if not np.all(np.isfinite(prob)):
        return {**out, **{m: NAN for m in METRICS}, "tn": -1, "fp": -1, "fn": -1, "tp": -1,
                "finite_outputs": False}
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    has_pos, has_neg = tp + fn > 0, tn + fp > 0
    out.update({
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0) if has_pos else NAN,
        "specificity": tn / (tn + fp) if has_neg else NAN,
        "fpr": fp / (fp + tn) if has_neg else NAN,
        "precision": precision_score(y_true, y_pred, zero_division=0) if (tp + fp) > 0 else NAN,
        "f1": f1_score(y_true, y_pred, zero_division=0) if has_pos else NAN,
        "roc_auc": roc_auc_score(y_true, prob) if (has_pos and has_neg) else NAN,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "finite_outputs": True,
    })
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in out.items()}


class Evaluator:
    """Evaluates a flat parameter vector on a fixed (X, y) set."""

    def __init__(self, X: np.ndarray, y: np.ndarray, model):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = np.asarray(y)
        self.yt = torch.from_numpy(self.y.astype(np.int64))
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def logits(self, params):
        set_flat_params(self.model, params)
        return self.model(self.X)

    @torch.no_grad()
    def predict_proba(self, params) -> np.ndarray:
        if len(self.y) == 0:
            return np.empty(0)
        return torch.softmax(self.logits(params), dim=1)[:, 1].numpy()

    @torch.no_grad()
    def quick(self, params) -> dict:
        """Per-round validation monitoring: CE loss, accuracy, malignant recall."""
        z = self.logits(params)
        loss = F.cross_entropy(z, self.yt).item()
        pred = z.argmax(1).numpy()
        pos = self.y == 1
        return {"loss": loss, "accuracy": float((pred == self.y).mean()),
                "recall": float((pred[pos] == 1).mean()) if pos.any() else NAN}

    def full(self, params) -> dict:
        return classification_metrics(self.y, self.predict_proba(params))
