TASK: A lending team wants to identify which new business financing requests are most likely to create losses so they can prioritize reviews, make faster and safer approval decisions, reduce bad debt, and allocate capital toward applicants with stronger repayment outlook while maintaining consistent, transparent decision quality across the portfolio.

-----------------------------------------------------

Business task: Ridge regression for B2B invoice default risk (tabular baseline)
Context: A fintech’s SMB lending team needs a fast, auditable baseline to score probability of 90-day default on new invoice-financing applications. Compliance wants stable, bounded coefficients (no wild swings when two highly correlated cash-flow metrics move together). Data science will later try richer models; first they need a production-safe linear baseline with strong regularization.

Data: A warehouse table (or CSV export) with one row per application: company age, sector dummies, recent revenue / burn proxies, debt-service ratios, bureau-style scores, counts of past late payments, macro factors, etc. Many columns are correlated (e.g. multiple revenue windows, similar ratio definitions).

Modeling: Train Ridge regression (or RidgeCV with CV over the penalty) on a 0/1 default label or on a continuous loss severity—your choice, but keep it linear + L2 only. Use a preprocessing pipeline (imputation, scaling) so penalties are meaningful. Tune 
α
α (or alphas in RidgeCV) via cross-validation.

Deliverables:

Holdout ROC-AUC / PR-AUC (if binary) or RMSE / R² (if continuous target), plus a decile lift or calibration-style table for ranking.
Coefficient report (after scaling): which factors increase vs decrease risk, with emphasis on stability across CV folds or a bootstrap (optional).
Short memo: why Ridge vs plain OLS here (collinearity, many features); latency (dot product after transform); retrain cadence; drift checks on top drivers.
Ethics: document excluded sensitive attributes; no future information (labels / outcomes after decision time).
This mirrors your Ridge note: many correlated inputs, stable coefficients, tabular risk baseline—without sparsity (that would be Lasso / Elastic Net).

## Code (this repo)

**Data proxy:** OpenML German Credit ([credit-g](https://www.openml.org/d/31)), `bad` vs `good`.

| File | What it does |
|------|----------------|
| `main.py` | Ridge + `GridSearchCV` for `alpha`; scaled numerics + one-hot; **`n_jobs=1`** so logs are not interleaved. |
| `main_boost.py` | Fixed-hyperparameter `GradientBoostingClassifier`, same holdout. |
| `main_advanced.py` | Extra numeric features, balanced `sample_weight`, `RandomizedSearchCV` tuned on PR-AUC. |

```bash
uv sync
uv run python main.py
uv run python main_boost.py
uv run python main_advanced.py
```

Use **`uv run`** from this directory so dependencies resolve in **this** project’s `.venv`. If another environment is active (e.g. a different prompt name), plain `py script.py` may hit **`ModuleNotFoundError: numpy`**. Fix: `deactivate`, then `uv run python …`, or `source .venv/Scripts/activate` and `python …` here.

Wrappers (always use this project’s env via `uv`): Git Bash `./run_main.sh`, `./run_boost.sh`, `./run_advanced.sh`; Windows CMD `run_advanced.bat`.

`main_advanced.py` also **re-runs itself with `uv run`** if `numpy` is missing (wrong active venv), so `py main_advanced.py` from this folder usually still works when `uv` is on `PATH`.