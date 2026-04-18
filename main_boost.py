import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def load_xy():
    bunch = fetch_openml(data_id=31, as_frame=True, parser="auto")
    X = bunch.data
    y_raw = bunch.target
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(np.float64).values
    else:
        y = (y_raw.astype(str).str.strip().str.lower() == "bad").astype(np.float64).values
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

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        random_state=42,
    )
    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("gb", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train_i = y_train.astype(np.int32)
    y_test_i = y_test.astype(np.int32)
    pipe.fit(X_train, y_train_i)
    y_score = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test_i, y_score)
    pr = average_precision_score(y_test_i, y_score)
    base = float(y_test_i.mean())
    n_test = len(y_test_i)
    k = max(1, n_test // 10)
    top_idx = np.argsort(-y_score)[:k]
    captured_pct = 100.0 * float(y_test_i[top_idx].sum()) / max(1.0, float(y_test_i.sum()))

    print("GradientBoostingClassifier (sklearn); same split as main.py (random_state=42)")
    print(f"data: {DATA_SOURCE}")
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

    gb = pipe.named_steps["gb"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = gb.feature_importances_
    order = np.argsort(imp)[::-1]
    print("feature_importances_ (top 20):")
    for j in order[:20]:
        if imp[j] <= 0:
            continue
        print(f"  {names[j]}: {imp[j]:.4f}")


if __name__ == "__main__":
    main()
