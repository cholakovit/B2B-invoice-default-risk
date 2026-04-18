import subprocess
import sys
from pathlib import Path


def _reexec_with_uv_if_needed() -> None:
    try:
        import numpy
    except ImportError:
        root = Path(__file__).resolve().parent
        script = Path(__file__).resolve()
        r = subprocess.run(
            ["uv", "run", "python", str(script), *sys.argv[1:]],
            cwd=root,
        )
        raise SystemExit(r.returncode)


_reexec_with_uv_if_needed()

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

DATA_SOURCE = "OpenML German Credit (credit-g); bad=1 / good=0 as default-risk proxy"


def decile_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-scores)
    ys = y_true[order]
    n = len(ys)
    rows = []
    for d in range(10):
        lo = d * n // 10
        hi = (d + 1) * n // 10 if d < 9 else n
        seg = ys[lo:hi]
        rows.append({
            "decile": d + 1,
            "n": hi - lo,
            "event_rate": float(seg.mean()) if len(seg) > 0 else 0.0,
            "events": int(seg.sum()),
        })
    return pd.DataFrame(rows)


def enrich_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "credit_amount" in X.columns and "duration" in X.columns:
        ca = pd.to_numeric(X["credit_amount"], errors="coerce")
        dur = pd.to_numeric(X["duration"], errors="coerce").clip(lower=1.0)
        X["AmountPerDuration"] = ca / dur
        X["LogCreditAmount"] = np.log1p(ca.fillna(0.0))
    if "duration" in X.columns:
        d = pd.to_numeric(X["duration"], errors="coerce").fillna(0.0)
        X["DurationSq"] = d ** 2
    if "age" in X.columns:
        a = pd.to_numeric(X["age"], errors="coerce").fillna(0.0)
        X["AgeSq"] = a ** 2
    return X


def load_xy():
    bunch = fetch_openml(data_id=31, as_frame=True, parser="auto")
    X = bunch.data
    y_raw = bunch.target
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(np.float64).values
    else:
        y = (y_raw.astype(str).str.strip().str.lower() == "bad").astype(np.float64).values
    X = enrich_features(X)
    return X, y


def main() -> None:
    X, y = load_xy()
    cat_cols = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "oh",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    preprocess = ColumnTransformer(transformers)

    base_gb = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("prep", preprocess), ("gb", base_gb)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train_i = y_train.astype(np.int32)
    y_test_i = y_test.astype(np.int32)
    sw = compute_sample_weight("balanced", y_train_i)

    search = RandomizedSearchCV(
        pipe,
        param_distributions={
            "gb__n_estimators": randint(80, 320),
            "gb__max_depth": randint(2, 9),
            "gb__learning_rate": uniform(0.02, 0.16),
            "gb__subsample": uniform(0.65, 0.30),
            "gb__min_samples_leaf": randint(1, 20),
            "gb__min_samples_split": randint(2, 18),
        },
        n_iter=28,
        cv=3,
        scoring="average_precision",
        random_state=42,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train_i, gb__sample_weight=sw)

    best = search.best_estimator_
    y_score = best.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test_i, y_score)
    pr = average_precision_score(y_test_i, y_score)
    base = float(y_test_i.mean())
    n_test = len(y_test_i)
    k = max(1, n_test // 10)
    top_idx = np.argsort(-y_score)[:k]
    captured_pct = 100.0 * float(y_test_i[top_idx].sum()) / max(1.0, float(y_test_i.sum()))

    print(
        "Tuned GBC + engineered numerics + balanced sample_weight; "
        "same split as main.py / main_boost.py (random_state=42)"
    )
    print(f"data: {DATA_SOURCE}")
    print(f"CV best mean PR-AUC: {search.best_score_:.4f}")
    print(f"holdout ROC-AUC: {auc:.4f}")
    print(f"holdout PR-AUC:  {pr:.4f}")
    print(
        f"decile lift (top 10% by risk score): capture {captured_pct:.1f}% of test defaults; "
        f"event rate in top decile {float(y_test_i[top_idx].mean()):.4f} vs base {base:.4f}"
    )
    print("decile event_rate (1 = highest predicted risk):")
    dt = decile_table(y_test_i.astype(np.float64), y_score)
    for _, r in dt.iterrows():
        print(f"  {int(r['decile'])}: rate={r['event_rate']:.4f} n={int(r['n'])}")
    print("best params:")
    for k1, v in sorted(search.best_params_.items()):
        print(f"  {k1}: {v}")

    gb = best.named_steps["gb"]
    names = best.named_steps["prep"].get_feature_names_out()
    imp = gb.feature_importances_
    order = np.argsort(imp)[::-1]
    print("feature_importances_ (top 20):")
    for j in order[:20]:
        if imp[j] <= 0:
            continue
        print(f"  {names[j]}: {imp[j]:.4f}")


if __name__ == "__main__":
    main()
