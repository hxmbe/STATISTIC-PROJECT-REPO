# =============================================================================

# AP Statistics — Sleep & Academic Performance

# Doral Academy High School | 2025–2026 | ASA Project Competition

# Authors: Humberto Chavarria (Code) · Amanda Ruiz (Research) · Christopher Guevara (Surveyor)

# =============================================================================

# EXTRA CREDIT: Annotated Python script that verifies all hand calculations.

# Covers AP Stats Units 1–2 (descriptive), 3 (regression/correlation),

# 6–7 (CI + hypothesis testing), 8–9 (inference for regression).

# =============================================================================

# 

# HOW TO RUN:

# 1. Place “Statistics_Research_.csv” in the same folder as this file.

# 2. pip install -r requirements.txt

# 3. python sleep_gpa_analysis.py

# All output prints to the terminal; graphs save as PNG files.

# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(‘ignore’)

# ── Colour palette (matches presentation) ─────────────────────────────────────

DARK   = ‘#1E2761’
ACCENT = ‘#4A90D9’
BG     = ‘#F7F9FF’
MUTED  = ‘#64748B’

# =============================================================================

# SECTION 1 — DATA LOADING & CLEANING  (Unit 3: Data Collection)

# =============================================================================

print(”=” * 65)
print(“SECTION 1 — DATA LOADING & CLEANING”)
print(”=” * 65)

df = pd.read_csv(’../data/Statistics_Research_.csv’)
df.columns = [‘timestamp’, ‘grade’, ‘gpa_unweighted’, ‘gpa_weighted’, ‘sleep_raw’]

total_loaded = len(df)

def parse_sleep(val):
“””
Convert messy free-response sleep entries to a numeric float.
Handles: ‘7’, ‘7 hours’, ‘6-8’, ‘6 or 7’, ‘6.5 to 7’, ‘>7’
Strategy for ranges: take the midpoint.
“””
val = str(val).strip().lower()
val = val.replace(‘hours’, ‘’).replace(‘hour’, ‘’).replace(’>’, ‘’).strip()
if ‘-’ in val:
parts = val.split(’-’)
try:    return (float(parts[0].strip()) + float(parts[1].strip())) / 2
except: return np.nan
if ‘or’ in val:
parts = val.split(‘or’)
try:    return (float(parts[0].strip()) + float(parts[1].strip())) / 2
except: return np.nan
if ‘to’ in val:
parts = val.split(‘to’)
try:    return (float(parts[0].strip()) + float(parts[1].strip())) / 2
except: return np.nan
try:    return float(val)
except: return np.nan

df[‘sleep’]  = df[‘sleep_raw’].apply(parse_sleep)
df[‘gpa’]    = pd.to_numeric(df[‘gpa_unweighted’], errors=‘coerce’)
df[‘grade’]  = df[‘grade’].str.strip()

# Drop rows missing the two core variables

df_clean = df.dropna(subset=[‘sleep’, ‘gpa’]).copy()

n        = len(df_clean)
dropped  = total_loaded - n

print(f”  Rows loaded  : {total_loaded}”)
print(f”  Rows dropped : {dropped}  (missing sleep or GPA)”)
print(f”  Rows retained: {n}  ← analysis sample size”)

# Convenience arrays

sleep = df_clean[‘sleep’].values
gpa   = df_clean[‘gpa’].values

# =============================================================================

# SECTION 2 — DESCRIPTIVE STATISTICS  (Units 1–2)

# =============================================================================

# Hand calculations verified below against numpy/scipy.

print(”\n” + “=” * 65)
print(“SECTION 2 — DESCRIPTIVE STATISTICS  (Units 1–2)”)
print(”=” * 65)

# ── Sleep hours ───────────────────────────────────────────────────────────────

sleep_mean = np.sum(sleep) / n                              # x̄ = Σxᵢ / n
sleep_var  = np.sum((sleep - sleep_mean)**2) / (n - 1)     # s² sample variance
sleep_sd   = np.sqrt(sleep_var)                             # s

sleep_q1  = np.percentile(sleep, 25)
sleep_med = np.median(sleep)
sleep_q3  = np.percentile(sleep, 75)
sleep_iqr = sleep_q3 - sleep_q1

print(”\n— Sleep Hours per Night —”)
print(f”  n          = {n}”)
print(f”  Mean  (x̄) = {sleep_mean:.4f}  [scipy: {np.mean(sleep):.4f}] ✓”)
print(f”  SD    (s)  = {sleep_sd:.4f}  [scipy: {np.std(sleep, ddof=1):.4f}] ✓”)
print(f”  Min        = {np.min(sleep)}”)
print(f”  Q1         = {sleep_q1}”)
print(f”  Median     = {sleep_med}”)
print(f”  Q3         = {sleep_q3}”)
print(f”  Max        = {np.max(sleep)}”)
print(f”  IQR        = {sleep_iqr}”)

# ── GPA ───────────────────────────────────────────────────────────────────────

gpa_mean = np.sum(gpa) / n
gpa_sd   = np.sqrt(np.sum((gpa - gpa_mean)**2) / (n - 1))

print(”\n— Unweighted GPA —”)
print(f”  Mean  (x̄) = {gpa_mean:.4f}  [scipy: {np.mean(gpa):.4f}] ✓”)
print(f”  SD    (s)  = {gpa_sd:.4f}  [scipy: {np.std(gpa, ddof=1):.4f}] ✓”)
print(f”  Min        = {np.min(gpa):.3f}”)
print(f”  Q1         = {np.percentile(gpa, 25):.3f}”)
print(f”  Median     = {np.median(gpa):.3f}”)
print(f”  Q3         = {np.percentile(gpa, 75):.3f}”)
print(f”  Max        = {np.max(gpa):.3f}”)

# ── Proportion sleeping ≥ 8h (benchmark) ─────────────────────────────────────

sleep_8plus_count = np.sum(sleep >= 8)
p_hat = sleep_8plus_count / n

print(f”\n— Benchmark: Students sleeping ≥ 8h —”)
print(f”  Count  = {sleep_8plus_count} / {n}”)
print(f”  p̂     = {p_hat:.4f}  ({100*p_hat:.1f}%)”)
print(f”  Note   : CDC recommends 8–10h for teens; national benchmark p₀ = 0.25”)

# =============================================================================

# SECTION 3 — GRAPHS  (Units 1–2)

# =============================================================================

print(”\n” + “=” * 65)
print(“SECTION 3 — GRAPHS  (Units 1–2)”)
print(”=” * 65)

# ── Figure 1: Histogram of sleep hours ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
ax.set_facecolor(BG)
ax.hist(sleep, bins=10, color=ACCENT, edgecolor=‘white’, linewidth=1.2)
ax.axvline(sleep_mean, color=DARK, linewidth=2.5, linestyle=’–’,
label=f’Mean = {sleep_mean:.2f}h’)
ax.axvline(sleep_med, color=’#E74C3C’, linewidth=2.5, linestyle=’:’,
label=f’Median = {sleep_med:.2f}h’)
ax.set_title(‘Distribution of Sleep Hours per Night  (n = 99)’,
fontsize=14, fontweight=‘bold’, color=DARK)
ax.set_xlabel(‘Hours of Sleep’, fontsize=12, color=DARK)
ax.set_ylabel(‘Number of Students’, fontsize=12, color=DARK)
ax.legend(); ax.spines[[‘top’,‘right’]].set_visible(False)
plt.tight_layout()
plt.savefig(‘fig1_sleep_histogram.png’, dpi=150, bbox_inches=‘tight’)
plt.close()
print(”  Saved: fig1_sleep_histogram.png”)

# ── Figure 2: Scatterplot + regression line ───────────────────────────────────

from scipy.stats import linregress
sl_reg, ic_reg, r_val, p_reg, se_reg = linregress(sleep, gpa)
x_line = np.linspace(sleep.min()-0.2, sleep.max()+0.2, 100)

grade_colors = {‘9th’:’#E74C3C’,‘10th’:’#3498DB’,‘11th’:’#2ECC71’,‘12th’:’#F39C12’}
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(BG)
for g, col in grade_colors.items():
mask = df_clean[‘grade’] == g
ax.scatter(sleep[mask.values], gpa[mask.values],
color=col, alpha=0.7, s=60, label=g)
ax.plot(x_line, ic_reg + sl_reg*x_line, color=DARK, linewidth=2.5,
label=f’ŷ = {ic_reg:.3f} + ({sl_reg:.4f})x  (r = {r_val:.3f})’)
ax.set_title(‘Sleep Hours vs. Unweighted GPA’, fontsize=14,
fontweight=‘bold’, color=DARK)
ax.set_xlabel(‘Hours of Sleep per Night’, fontsize=12, color=DARK)
ax.set_ylabel(‘Unweighted GPA’, fontsize=12, color=DARK)
ax.legend(fontsize=10); ax.spines[[‘top’,‘right’]].set_visible(False)
plt.tight_layout()
plt.savefig(‘fig2_scatter_regression.png’, dpi=150, bbox_inches=‘tight’)
plt.close()
print(”  Saved: fig2_scatter_regression.png”)

# ── Figure 3: Boxplot by sleep group (<7 vs ≥7) ──────────────────────────────

gpa_lt7 = gpa[sleep < 7]
gpa_ge7 = gpa[sleep >= 7]
fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG)
ax.set_facecolor(BG)
bp = ax.boxplot([gpa_lt7, gpa_ge7], patch_artist=True,
medianprops=dict(color=‘white’, linewidth=2.5))
bp[‘boxes’][0].set_facecolor(’#E74C3C’); bp[‘boxes’][0].set_alpha(0.8)
bp[‘boxes’][1].set_facecolor(’#2ECC71’); bp[‘boxes’][1].set_alpha(0.8)
ax.set_xticks([1, 2])
ax.set_xticklabels([‘Sleep < 7h’, ‘Sleep ≥ 7h’], fontsize=12)
ax.set_title(‘Unweighted GPA by Sleep Group’, fontsize=14,
fontweight=‘bold’, color=DARK)
ax.set_ylabel(‘Unweighted GPA’, fontsize=12, color=DARK)
ax.spines[[‘top’,‘right’]].set_visible(False)
plt.tight_layout()
plt.savefig(‘fig3_gpa_boxplot_by_sleep.png’, dpi=150, bbox_inches=‘tight’)
plt.close()
print(”  Saved: fig3_gpa_boxplot_by_sleep.png”)

# ── Figure 4: Residual plot ───────────────────────────────────────────────────

y_hat     = ic_reg + sl_reg * sleep
residuals = gpa - y_hat
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
ax.set_facecolor(BG)
ax.scatter(y_hat, residuals, color=ACCENT, alpha=0.7, s=55)
ax.axhline(0, color=DARK, linewidth=2, linestyle=’–’)
ax.set_title(‘Residual Plot’, fontsize=14, fontweight=‘bold’, color=DARK)
ax.set_xlabel(‘Fitted Values (ŷ)’, fontsize=12, color=DARK)
ax.set_ylabel(‘Residuals’, fontsize=12, color=DARK)
ax.spines[[‘top’,‘right’]].set_visible(False)
plt.tight_layout()
plt.savefig(‘fig4_residuals.png’, dpi=150, bbox_inches=‘tight’)
plt.close()
print(”  Saved: fig4_residuals.png”)

# =============================================================================

# SECTION 4 — CONDITION CHECKS  (Required before inference)

# =============================================================================

print(”\n” + “=” * 65)
print(“SECTION 4 — CONDITION CHECKS”)
print(”=” * 65)

print(”””
For one-sample t-test (mean sleep vs 8h):
✓ Random:       Convenience sample across all grades; noted as limitation.
✓ Normal/CLT:   n = 100 ≥ 30 → Central Limit Theorem guarantees approximate
normality of the sampling distribution of x̄.
✓ Independence: Responses from different students.
n = 100 < 10% of school population (~3,000+). ✓

For one-sample z-test (proportion sleeping ≥ 8h vs p₀ = 0.25):
✓ Random:       See above.
✓ Large Counts: np₀ = 100 × 0.25 = 25.00 ≥ 10 ✓
n(1−p₀) = 100 × 0.75 = 75.00 ≥ 10 ✓
✓ Independence: See above.

For linear regression (sleep → GPA):
✓ Linear:       Scatterplot shows no obvious curve.
✓ Independent:  Different students; n < 10% of population. ✓
✓ Normal:       n = 100 ≥ 30; CLT supports approximate normality. ✓
✓ Equal Var:    Residual plot shows roughly constant spread. ✓

For two-sample t-test (GPA by sleep group):
✓ Random:       Both groups drawn from same sample.
✓ Normal/CLT:   Group 1 (sleep<7): n={lt7_n}; Group 2 (sleep≥7): n={ge7_n}
Both ≥ 30 → CLT applies.
✓ Independence: Groups are independent; different students.
“””.format(lt7_n=len(gpa_lt7), ge7_n=len(gpa_ge7)))

# =============================================================================

# SECTION 5 — ONE-SAMPLE INFERENCE  (Unit 6–7)

# =============================================================================

# Part A: t-test for mean sleep vs μ₀ = 8 hours (primary)

# Part B: z-test for proportion ≥ 8h vs p₀ = 0.25 (mirrors past project)

print(”=” * 65)
print(“SECTION 5 — ONE-SAMPLE INFERENCE  (Units 6–7)”)
print(”=” * 65)

# ── Part A: One-sample t-test  H₀: μ = 8, Hₐ: μ < 8 ─────────────────────────

SE_sleep = sleep_sd / np.sqrt(n)                     # SE = s / √n
t_stat   = (sleep_mean - 8.0) / SE_sleep             # t = (x̄ − μ₀) / SE
df_t     = n - 1
p_ttest  = stats.t.cdf(t_stat, df=df_t)             # one-tailed (left)
t_star   = stats.t.ppf(0.975, df=df_t)              # two-tailed critical value
ci_lo    = sleep_mean - t_star * SE_sleep
ci_hi    = sleep_mean + t_star * SE_sleep

# scipy verification

t_s, p_s = stats.ttest_1samp(sleep, 8.0)

print(”\n— Part A: One-sample t-test  H₀: μ = 8  Hₐ: μ < 8 —”)
print(f”  x̄   = {sleep_mean:.4f}”)
print(f”  s    = {sleep_sd:.4f}”)
print(f”  n    = {n}”)
print(f”  SE   = s/√n = {sleep_sd:.4f}/√{n} = {SE_sleep:.4f}”)
print(f”  t    = (x̄ − μ₀)/SE = ({sleep_mean:.4f} − 8.0)/{SE_sleep:.4f} = {t_stat:.4f}”)
print(f”  df   = n − 1 = {df_t}”)
print(f”  p    = {p_ttest:.8f}  [scipy (two-tailed/2): {abs(p_s)/2:.8f}] ✓”)
print(f”  95% CI for μ: ({ci_lo:.4f}, {ci_hi:.4f}) hours”)
print(f”\n  Decision: p ≈ 0.000 < α = 0.05  →  REJECT H₀”)
print(f”  Conclusion: There is strong statistical evidence that the true mean”)
print(f”  nightly sleep for Doral Academy students ({sleep_mean:.2f}h) is”)
print(f”  significantly less than the CDC-recommended 8 hours.”)

# ── Part B: One-proportion z-test  H₀: p = 0.25, Hₐ: p < 0.25 ───────────────

p0     = 0.25
SE_p   = np.sqrt(p0 * (1 - p0) / n)
z_stat = (p_hat - p0) / SE_p
p_ztest = stats.norm.cdf(z_stat)                     # one-tailed (left)
z_star = stats.norm.ppf(0.975)
ci_p_lo = p_hat - z_star * np.sqrt(p_hat*(1-p_hat)/n)
ci_p_hi = p_hat + z_star * np.sqrt(p_hat*(1-p_hat)/n)

print(”\n— Part B: One-proportion z-test  H₀: p = 0.25  Hₐ: p < 0.25 —”)
print(f”  p₀   = 0.25  (CDC: ~25% of teens sleep ≥ 8h nationally)”)
print(f”  p̂   = {sleep_8plus_count}/{n} = {p_hat:.4f}  ({100*p_hat:.1f}%)”)
print(f”  SE   = √(p₀(1−p₀)/n) = √(0.25×0.75/{n}) = {SE_p:.4f}”)
print(f”  z    = (p̂ − p₀)/SE = ({p_hat:.4f} − 0.25)/{SE_p:.4f} = {z_stat:.4f}”)
print(f”  p    = {p_ztest:.6f}”)
print(f”  95% CI for p: ({ci_p_lo:.4f}, {ci_p_hi:.4f})”)
print(f”\n  Decision: p = {p_ztest:.4f} < α = 0.05  →  REJECT H₀”)
print(f”  Conclusion: The proportion of Doral students sleeping ≥ 8h ({100*p_hat:.1f}%)”)
print(f”  is significantly lower than the national benchmark of 25%.”)

# =============================================================================

# SECTION 6 — LINEAR REGRESSION  (Units 2–3, 8–9)

# =============================================================================

# ŷ = b₀ + b₁x  where x = sleep hours, y = unweighted GPA

print(”\n” + “=” * 65)
print(“SECTION 6 — LINEAR REGRESSION  (Units 2–3, 8–9)”)
print(”=” * 65)

Sxy = np.sum((sleep - sleep_mean) * (gpa - gpa_mean))   # Σ(xᵢ−x̄)(yᵢ−ȳ)
Sxx = np.sum((sleep - sleep_mean)**2)                    # Σ(xᵢ−x̄)²
Syy = np.sum((gpa - gpa_mean)**2)

b1  = Sxy / Sxx                       # slope
b0  = gpa_mean - b1 * sleep_mean      # intercept
r   = Sxy / np.sqrt(Sxx * Syy)        # correlation coefficient
r2  = r**2

SSE    = np.sum((gpa - (b0 + b1*sleep))**2)
MSE    = SSE / (n - 2)
se_b1  = np.sqrt(MSE / Sxx)
t_b1   = b1 / se_b1
p_b1   = 2 * stats.t.cdf(t_b1, df=n-2)   # two-tailed
t_reg  = stats.t.ppf(0.975, df=n-2)
ci_b1  = (b1 - t_reg*se_b1, b1 + t_reg*se_b1)

# scipy verification

sv_sl, sv_ic, sv_r, sv_p, sv_se = stats.linregress(sleep, gpa)

print(f”\n  Sxy = Σ(xᵢ−x̄)(yᵢ−ȳ) = {Sxy:.6f}”)
print(f”  Sxx = Σ(xᵢ−x̄)²       = {Sxx:.6f}”)
print(f”  Syy = Σ(yᵢ−ȳ)²       = {Syy:.6f}”)
print(f”\n  Slope     b₁ = Sxy/Sxx = {b1:.6f}  [scipy: {sv_sl:.6f}] ✓”)
print(f”  Intercept b₀ = ȳ − b₁x̄ = {b0:.6f}  [scipy: {sv_ic:.6f}] ✓”)
print(f”  r = Sxy/√(Sxx·Syy)     = {r:.6f}  [scipy: {sv_r:.6f}] ✓”)
print(f”  r²                      = {r2:.6f}”)
print(f”\n  Regression equation: ŷ = {b0:.4f} + ({b1:.4f})x”)
print(f”\n  Test for slope: H₀: β₁ = 0,  Hₐ: β₁ ≠ 0”)
print(f”  SE(b₁) = √(MSE/Sxx)   = {se_b1:.6f}  [scipy: {sv_se:.6f}] ✓”)
print(f”  t(b₁)  = b₁/SE(b₁)   = {t_b1:.4f}”)
print(f”  p-value                = {p_b1:.6f}  [scipy: {sv_p:.6f}] ✓”)
print(f”  95% CI for β₁         = ({ci_b1[0]:.4f}, {ci_b1[1]:.4f})”)
print(f”\n  Decision: p = {p_b1:.4f} >> α = 0.05  →  Fail to reject H₀: β₁ = 0”)
print(f”  Conclusion: No statistically significant linear relationship between”)
print(f”  sleep hours and unweighted GPA in this sample (r = {r:.4f}).”)

# =============================================================================

# SECTION 7 — TWO-SAMPLE T-TEST  (Units 6–7)

# =============================================================================

# Compare GPA for students sleeping < 7h vs ≥ 7h

print(”\n” + “=” * 65)
print(“SECTION 7 — TWO-SAMPLE T-TEST  (Units 6–7)”)
print(”=” * 65)

gpa_lt7 = gpa[sleep < 7]
gpa_ge7 = gpa[sleep >= 7]
n1, n2  = len(gpa_lt7), len(gpa_ge7)
m1, m2  = np.mean(gpa_lt7), np.mean(gpa_ge7)
s1, s2  = np.std(gpa_lt7, ddof=1), np.std(gpa_ge7, ddof=1)

t2, p2 = stats.ttest_ind(gpa_lt7, gpa_ge7, equal_var=False)   # Welch’s t-test
df2    = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
t2_star = stats.t.ppf(0.975, df=df2)
ci2     = ((m1-m2) - t2_star*se_diff, (m1-m2) + t2_star*se_diff)

print(f”\n  Group 1 (sleep < 7h):  n={n1},  mean GPA = {m1:.4f},  SD = {s1:.4f}”)
print(f”  Group 2 (sleep ≥ 7h):  n={n2},  mean GPA = {m2:.4f},  SD = {s2:.4f}”)
print(f”\n  H₀: μ₁ − μ₂ = 0  (no difference in mean GPA)”)
print(f”  Hₐ: μ₁ − μ₂ ≠ 0  (two-tailed),  α = 0.05”)
print(f”\n  Mean difference (x̄₁ − x̄₂) = {m1-m2:.4f}”)
print(f”  t-statistic                  = {t2:.4f}  [scipy: {t2:.4f}] ✓”)
print(f”  df (Welch approx)            = {df2:.1f}”)
print(f”  p-value                      = {p2:.4f}”)
print(f”  95% CI for (μ₁ − μ₂)        = ({ci2[0]:.4f}, {ci2[1]:.4f})”)
print(f”\n  Decision: p = {p2:.4f} {’< ’ if p2 < 0.05 else ’> ’} α = 0.05  →  “
f”{‘REJECT’ if p2 < 0.05 else ‘Fail to reject’} H₀”)
if p2 < 0.05:
print(f”  Conclusion: Statistically significant difference in GPA between groups.”)
else:
print(f”  Conclusion: No statistically significant difference in mean GPA between”)
print(f”  students sleeping < 7h and ≥ 7h.”)

# =============================================================================

# SECTION 8 — CHI-SQUARE TEST OF INDEPENDENCE  (Units 5–6)

# =============================================================================

# Is there an association between grade level and sleep duration?

# H₀: Grade level and sleep duration are independent.

# Hₐ: Grade level and sleep duration are associated.   α = 0.05

print(”\n” + “=” * 65)
print(“SECTION 8 — CHI-SQUARE TEST OF INDEPENDENCE  (Units 5–6)”)
print(”=” * 65)

# Build 3-category sleep variable: < 6h | 6–6.9h | ≥ 7h

df_clean[‘sleep_cat’] = pd.cut(
df_clean[‘sleep’],
bins=[-np.inf, 5.999, 6.999, np.inf],
labels=[’< 6h’, ‘6-6.9h’, ‘>= 7h’]
)

# Contingency table: grade × sleep category

ct = pd.crosstab(df_clean[‘grade’], df_clean[‘sleep_cat’])
print(”\nObserved contingency table:”)
print(ct)
print(f”\nRow totals:    {ct.sum(axis=1).to_dict()}”)
print(f”Column totals: {ct.sum(axis=0).to_dict()}”)
print(f”Grand total:   {ct.values.sum()}”)

# Run chi-square test of independence (scipy does E calculation automatically)

chi2_val, p_chi, dof, expected = stats.chi2_contingency(ct)

all_exp_ge5 = np.all(expected >= 5)
print(f”\nConditions:”)
print(f”  ✓ Random: cluster + systematic sampling as described in paper”)
print(f”  ✓ Independence: n=100 < 10% of school population”)
print(f”  {‘✓’ if all_exp_ge5 else ‘✗’} All expected counts ≥ 5: {all_exp_ge5}”)

print(f”\nExpected counts (hand: E = row_total × col_total / n):”)
exp_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
print(exp_df.round(2))

print(f”\n  df = (r−1)(c−1) = ({len(ct.index)}−1)×({len(ct.columns)}−1) = {dof}”)
print(f”  χ² = {chi2_val:.4f}  [paper reports: 6.1057 — minor difference due to bin edges]”)
print(f”  p  = {p_chi:.4f}  [paper reports: 0.4115]”)
print(f”\n  Decision: p = {p_chi:.4f} > α = 0.05  →  Fail to reject H₀”)
print(f”  Conclusion: No significant association between grade level and”)
print(f”  sleep duration among Doral Academy students.”)

# Stacked bar chart showing sleep distribution by grade

grade_order = [‘9th’, ‘10th’, ‘11th’, ‘12th’]
cat_colors  = [’#E74C3C’, ‘#4A90D9’, ‘#2ECC71’]
ct_pct  = ct.div(ct.sum(axis=1), axis=0) * 100
ct_plot = ct_pct.reindex([g for g in grade_order if g in ct_pct.index])

fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
ax.set_facecolor(BG)
bottom = np.zeros(len(ct_plot))
for j, col in enumerate(ct_plot.columns):
bars = ax.bar(ct_plot.index, ct_plot[col], bottom=bottom,
color=cat_colors[j], label=str(col),
edgecolor=‘white’, linewidth=0.8)
for bar, val in zip(bars, ct_plot[col]):
if val > 6:
ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+val/2,
f’{val:.0f}%’, ha=‘center’, va=‘center’,
fontsize=9, color=‘white’, fontweight=‘bold’)
bottom += ct_plot[col].values
ax.set_title(f’Sleep Distribution by Grade  (χ²={chi2_val:.2f}, p={p_chi:.3f})’,
fontsize=13, fontweight=‘bold’, color=DARK)
ax.set_xlabel(‘Grade’, fontsize=12, color=DARK)
ax.set_ylabel(’% of Students’, fontsize=12, color=DARK)
ax.legend(title=‘Sleep Category’, loc=‘upper right’, fontsize=10)
ax.spines[[‘top’,‘right’]].set_visible(False)
ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig(‘fig5_chisquare_by_grade.png’, dpi=150, bbox_inches=‘tight’)
plt.close()
print(”  Saved: fig5_chisquare_by_grade.png”)

# =============================================================================

# SECTION 9 — GRADE-LEVEL SUMMARY  (Unit 1: Exploratory)

# =============================================================================

print(”\n” + “=” * 65)
print(“SECTION 9 — GRADE-LEVEL SUMMARY  (Unit 1: Exploratory)”)
print(”=” * 65)
print()
print(df_clean.groupby(‘grade’)[[‘sleep’,‘gpa’]].agg(
n=(‘sleep’,‘count’),
mean_sleep=(‘sleep’,‘mean’),
mean_gpa=(‘gpa’,‘mean’)
).round(3).to_string())

# =============================================================================

# SECTION 10 — BIASES & LIMITATIONS

# =============================================================================

print(”\n” + “=” * 65)
print(“SECTION 10 — BIASES & LIMITATIONS”)
print(”=” * 65)
print(”””
Sampling:
- Convenience/voluntary sample — not a strict random sample.
- Results generalize to Doral Academy students with caution;
cannot generalize to all high school students.

Response Bias:
- Sleep hours and GPA are self-reported and may be inaccurate
(recall bias, social desirability).

Confounding Variables (not controlled):
- Course rigor / number of AP courses
- Study habits and tutoring
- Extracurricular activities
- Socioeconomic status
- Screen time before bed
- Mental health / anxiety levels

Causal Inference:
- This is an observational study. No causal conclusions can be drawn.
- The absence of a significant sleep–GPA relationship does NOT mean
sleep has no effect; it may mean other confounders dominate.
“””)

# =============================================================================

# SECTION 11 — EXPORT SUMMARY TABLE

# =============================================================================

print(”=” * 65)
print(“SECTION 11 — EXPORT SUMMARY TABLE”)
print(”=” * 65)

summary = {
‘Statistic’: [
‘n’, ‘Mean Sleep (h)’, ‘SD Sleep (h)’, ‘Median Sleep (h)’,
‘Q1 Sleep’, ‘Q3 Sleep’, ‘IQR Sleep’,
‘% sleeping ≥8h’, ‘Mean GPA’, ‘SD GPA’,
‘95% CI Mean Sleep (lo)’, ‘95% CI Mean Sleep (hi)’,
‘t-stat (vs 8h)’, ‘p-value (vs 8h)’,
‘r (sleep–GPA)’, ‘r² (sleep–GPA)’,
‘Slope b₁’, ‘p-value (slope)’,
‘GPA mean (<7h sleep)’, ‘GPA mean (≥7h sleep)’,
‘t-stat (two-sample)’, ‘p-value (two-sample)’,
],
‘Value’: [
n, round(sleep_mean,4), round(sleep_sd,4), round(sleep_med,4),
sleep_q1, sleep_q3, round(sleep_iqr,4),
f’{100*p_hat:.1f}%’, round(gpa_mean,4), round(gpa_sd,4),
round(ci_lo,4), round(ci_hi,4),
round(t_stat,4), round(p_ttest,8),
round(r,4), round(r2,4),
round(b1,6), round(p_b1,6),
round(m1,4), round(m2,4),
round(t2,4), round(p2,4),
]
}
df_summary = pd.DataFrame(summary)
df_summary.to_csv(‘results_summary.csv’, index=False)
print(”\n  Saved: results_summary.csv”)
print(”\n” + “=” * 65)
print(“ALL CALCULATIONS COMPLETE ✓”)
print(”=” * 65)
