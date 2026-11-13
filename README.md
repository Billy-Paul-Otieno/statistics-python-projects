# Statistics Projects in Python

Two short, reproducible projects showing how I approach statistical problems: state hypotheses, check assumptions, run the test, quantify the effect, and verify the result a second way.

> **Note:** Data is **simulated** with fixed random seeds, so every run gives identical results. The goal is to demonstrate method, not to claim real-world findings.

## Setup
```bash
pip install -r requirements.txt
python hypothesis_testing.py
python regression_analysis.py
```

## 1. Hypothesis testing (`hypothesis_testing.py`)
**Question:** Do students who attend revision sessions score higher than those who don't?

**Method**
1. State H0 and H1 (two-sided, alpha = 0.05)
2. Check assumptions: Shapiro-Wilk (normality), Levene (equal variances)
3. Welch's two-sample t-test
4. Report effect size (Cohen's d) and a 95% confidence interval
5. Cross-check with the non-parametric Mann-Whitney U test

**Result (seed 42, n = 40 per group)**
| Metric | Value |
|---|---|
| Welch t / p | 3.014 / 0.0035 |
| Mean difference | 5.25 marks (95% CI 1.78 to 8.72) |
| Cohen's d | 0.67 (medium-large) |
| Mann-Whitney p | 0.0066 |

Both tests agree, so H0 is rejected at the 5% level.

## 2. Linear regression from scratch (`regression_analysis.py`)
**Model:** `score = b0 + b1*study_hours + b2*sleep_hours + noise`

**Method**
- Simple regression with `scipy.stats.linregress`
- Multiple regression via the normal equations, `b = (X'X)^-1 X'y`, with standard errors, t-values, and p-values computed by hand
- **Verification:** coefficients asserted equal to `numpy.linalg.lstsq`
- Residual diagnostics (mean near 0, Shapiro-Wilk, residuals-vs-fitted plot)

**Result (seed 7, n = 120)**
| Term | Coefficient | p-value |
|---|---|---|
| intercept | 32.00 | < 0.001 |
| study_hours | 4.12 | < 0.001 |
| sleep_hours | 1.58 | < 0.001 |

R² = 0.828, adjusted R² = 0.826. Plots are saved as `regression_plots.png` when you run the script.

## What I'd do next
- Repeat on a real public dataset
- Add multiple-testing correction and power analysis
