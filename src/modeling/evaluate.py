"""
Model evaluation utilities.

Reports Accuracy, Precision, Recall, F1, ROC-AUC, and Precision-Recall AUC.
PR-AUC is included alongside ROC-AUC because ROC-AUC can look misleadingly
strong under class imbalance (methodology Section 5.4).
"""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate_classifier(model, X_test, y_test, name: str = "model") -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    print(f"[{name}] " + " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["benign", "pathogenic"]))

    return metrics
