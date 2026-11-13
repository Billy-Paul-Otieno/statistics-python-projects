"""
Simple and multiple linear regression from scratch (normal equations),
verified against scipy.  Data is SIMULATED (seeded).

Model: score = b0 + b1*study_hours + b2*sleep_hours + noise

Run:  python regression_analysis.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(7)
n = 120
study = rng.uniform(0, 10, n)
sleep = rng.uniform(4, 9, n)
score = 30 + 4.0 * study + 2.0 * sleep + rng.normal(0, 6, n)

# ---- Simple regression: score ~ study ------------------------------------
res = stats.linregress(study, score)
print("Simple regression (score ~ study_hours)")
print(f"  slope = {res.slope:.3f}, intercept = {res.intercept:.3f}")
print(f"  R^2 = {res.rvalue**2:.3f}, p(slope) = {res.pvalue:.2e}")

# ---- Multiple regression via normal equations: b = (X'X)^-1 X'y ---------
X = np.column_stack([np.ones(n), study, sleep])
beta = np.linalg.solve(X.T @ X, X.T @ score)
resid = score - X @ beta
k = X.shape[1]
sigma2 = resid @ resid / (n - k)
cov = sigma2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
tvals = beta / se
pvals = 2 * stats.t.sf(np.abs(tvals), df=n - k)
r2 = 1 - (resid @ resid) / np.sum((score - score.mean()) ** 2)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

print("\nMultiple regression (normal equations)")
for name, b, s, p in zip(["intercept", "study_hours", "sleep_hours"], beta, se, pvals):
    print(f"  {name:12s} coef = {b:7.3f}  se = {s:6.3f}  p = {p:.2e}")
print(f"  R^2 = {r2:.3f}, adjusted R^2 = {adj_r2:.3f}")

# ---- Verification: matches numpy's least squares ---------------------------
beta_np, *_ = np.linalg.lstsq(X, score, rcond=None)
assert np.allclose(beta, beta_np), "Normal equations disagree with lstsq!"
print("\nCheck passed: coefficients match numpy.linalg.lstsq")

# ---- Residual diagnostics ------------------------------------------------------
print(f"Residual mean = {resid.mean():.2e} (should be ~0)")
print(f"Shapiro-Wilk on residuals: p = {stats.shapiro(resid).pvalue:.3f}")

# ---- Plots --------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(study, score, alpha=0.6)
xs = np.linspace(0, 10, 50)
ax[0].plot(xs, res.intercept + res.slope * xs, color="red")
ax[0].set(xlabel="Study hours", ylabel="Score", title="Simple regression")
ax[1].scatter(X @ beta, resid, alpha=0.6)
ax[1].axhline(0, color="red")
ax[1].set(xlabel="Fitted", ylabel="Residual", title="Residuals vs fitted")
plt.tight_layout()
plt.savefig("regression_plots.png", dpi=120)
print("Saved regression_plots.png")
