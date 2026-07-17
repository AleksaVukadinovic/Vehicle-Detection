from __future__ import annotations
import json
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.config import TrainConfig
from src.dataset import get_numpy_arrays


def evaluate_classifier(name, clf, x_test, y_test) -> dict:
    proba = clf.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def main() -> None:
    cfg = TrainConfig()
    print("Loading CIFAR-10 as flat numpy arrays...")
    x_train, y_train, x_test, y_test = get_numpy_arrays(cfg)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    # Standardize features (fit on train only).
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    results = []

    print("\nTraining Logistic Regression baseline...")
    start = time.time()
    logreg = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1)
    logreg.fit(x_train_s, y_train)
    logreg_time = time.time() - start
    r = evaluate_classifier("Logistic Regression", logreg, x_test_s, y_test)
    r["train_time_s"] = round(logreg_time, 1)
    results.append(r)
    print(f"  accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}  ({logreg_time:.1f}s)")

    print("\nTraining MLP (fully-connected, no convolutions) baseline...")
    start = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=40,
        early_stopping=True,
        random_state=cfg.seed,
    )
    mlp.fit(x_train_s, y_train)
    mlp_time = time.time() - start
    r = evaluate_classifier("MLP (no conv)", mlp, x_test_s, y_test)
    r["train_time_s"] = round(mlp_time, 1)
    results.append(r)
    print(f"  accuracy={r['accuracy']:.4f}  f1={r['f1']:.4f}  ({mlp_time:.1f}s)")

    out_path = cfg.checkpoint_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline results to {out_path}")


if __name__ == "__main__":
    main()
