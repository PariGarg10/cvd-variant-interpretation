"""
Model evaluation utilities.

Reports Accuracy, Precision, Recall, Specificity, F1, balanced accuracy,
Matthews correlation, ROC-AUC and Precision-Recall AUC. PR-AUC is included
alongside ROC-AUC because ROC-AUC can look misleadingly strong under class
imbalance, and specificity/balanced accuracy because a model that calls
almost everything pathogenic can still have high recall and F1.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "prevalence": float(y_true.mean()),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def bootstrap_ci(y_true, scores, metric, n_boot: int = 1000, seed: int = 42) -> list[float]:
    """Percentile 95% bootstrap confidence interval for a ranking metric."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric(y_true[idx], scores[idx]))
    return [round(float(np.percentile(values, 2.5)), 3), round(float(np.percentile(values, 97.5)), 3)]


def evaluate_classifier(model, X_test, y_test, name: str = "model") -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, proba)
    print(f"[{name}] " + " | ".join(
        f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float)
    ))
    return metrics
