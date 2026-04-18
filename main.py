import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def main() -> None:
    bunch = fetch_openml(data_id=31, as_frame=True, parser="auto")
    X = bunch.data
    y_raw = bunch.target
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(np.float64).values
    else:
        y = (y_raw.astype(str).str.strip().str.lower() == "bad").astype(np.float64).values

    cat_cols = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
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

    alphas = np.logspace(-3, 4, 60)
    pipe = Pipeline([("prep", preprocess), ("ridge", Ridge())])
    search = GridSearchCV(
        pipe,
        {"ridge__alpha": alphas},
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        refit=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    y_score = best.predict(X_test)

    auc = roc_auc_score(y_test, y_score)
    pr = average_precision_score(y_test, y_score)
    base = float(y_test.mean())
    n_test = len(y_test)
    k = max(1, n_test // 10)
    top_idx = np.argsort(-y_score)[:k]
    captured_pct = 100.0 * float(y_test[top_idx].sum()) / max(1.0, float(y_test.sum()))

    print(f"data: {DATA_SOURCE}")
    print(f"CV best alpha: {search.best_params_['ridge__alpha']:.6f}")
    print(f"holdout ROC-AUC: {auc:.4f}")
    print(f"holdout PR-AUC:  {pr:.4f}")
    print(
        f"decile lift (top 10% by risk score): capture {captured_pct:.1f}% of test defaults; "
        f"event rate in top decile {float(y_test[top_idx].mean()):.4f} vs base {base:.4f}"
    )
    print("decile event_rate (1 = highest predicted risk):")
    dt = decile_table(y_test, y_score)
    for _, r in dt.iterrows():
        print(f"  {int(r['decile'])}: rate={r['event_rate']:.4f} n={int(r['n'])}")

    ridge = best.named_steps["ridge"]
    names = best.named_steps["prep"].get_feature_names_out()
    coef = ridge.coef_
    order = np.argsort(np.abs(coef))[::-1]
    print("coefficients (linear risk score; higher ~ higher default risk):")
    for j in order[:30]:
        if abs(coef[j]) < 1e-12:
            continue
        print(f"  {names[j]}: {coef[j]:+.6f}")
    print(f"  intercept: {ridge.intercept_:+.6f}")
    print(
        "deployment: fixed preprocessing then dot product; sub-ms per row at scale; "
        "retrain on schedule or when PSI/drift on top features exceeds policy."
    )
    print(
        "ethics: exclude protected attributes where required; prove decision-time cutoff "
        "so labels and post-decision fields do not leak into X."
    )


if __name__ == "__main__":
    main()