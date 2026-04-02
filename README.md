# Sleep & Academic Performance — AP Statistics
**Doral Academy High School · AP Statistics · 2025–2026**  
**Authors: Humberto Chavarria · Amanda Ruiz · Christopher Guevara**  
**Instructor: Prof. Kristina Zogović**

---

## Research Question
> What is the mean number of hours students at Doral Academy sleep per night, and is there a significant linear relationship between sleep hours and unweighted GPA?

---

## Key Results

| Finding | Result |
|---|---|
| Mean sleep / night | **6.42h** (95% CI: 6.20–6.64h) |
| vs. CDC 8h benchmark | t = −14.48, **p ≈ 0.000** → Reject H₀ |
| % sleeping ≥ 8h | **12.1%** (vs. 25% national — p = 0.0015) |
| Sleep–GPA correlation | r = −0.011, **p = 0.917** → No significant relationship |
| Two-sample GPA test | p = 0.978 → No significant GPA difference by sleep group |

---

## Repository Structure

```
sleep-gpa-stats/
├── README.md
├── data/
│   └── Statistics_Research_.csv      ← raw survey data (upload this yourself)
├── analysis/
│   ├── sleep_gpa_analysis.py         ← annotated Python script (extra credit)
│   └── requirements.txt
└── dashboard/
    ├── index.html                     ← interactive web dashboard
    ├── fig1_sleep_histogram.png
    ├── fig2_scatter_regression.png
    ├── fig3_gpa_boxplot_by_sleep.png
    ├── fig4_residuals.png
    └── results_summary.csv
```

---

## How to Run the Analysis

### Option 1 — Run locally
```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/sleep-gpa-stats.git
cd sleep-gpa-stats

# 2. Install dependencies
pip install -r analysis/requirements.txt

# 3. Make sure Statistics_Research_.csv is in the data/ folder, then:
cd analysis
python sleep_gpa_analysis.py
```

### Option 2 — Run on Replit (no install needed)
1. Go to **[replit.com](https://replit.com)**
2. Click **Create Repl → Import from GitHub**
3. Paste this repo URL
4. Upload `Statistics_Research_.csv` to the `data/` folder
5. Set the run command to: `python analysis/sleep_gpa_analysis.py`
6. Hit **Run ▶**

### Option 3 — View Interactive Dashboard
Enable **GitHub Pages** (Settings → Pages → Source: main → /dashboard)  
Dashboard will be live at: `https://YOUR_USERNAME.github.io/sleep-gpa-stats/`

---

## AP Statistics Units Covered

| Unit | Topic | Where Used |
|---|---|---|
| Unit 1 | Exploring One-Variable Data | Histograms, boxplots, five-number summary |
| Unit 2 | Exploring Two-Variable Data | Scatterplot, regression, correlation |
| Unit 3 | Collecting Data | Survey design, sampling method |
| Unit 6 | Inference for Categorical Data | One-proportion z-test (p̂ vs 0.25) |
| Unit 7 | Inference for Quantitative Data | One-sample t-test, two-sample t-test |
| Unit 8 | Inference for Regression | Test for slope β₁, CI for slope |

---

## What the Script Verifies (Extra Credit)

All statistics are **computed by hand formula first**, then verified against `scipy`:

- ✅ Mean, variance, SD (x̄ = Σxᵢ/n ; s² = Σ(xᵢ−x̄)²/(n−1))
- ✅ Five-number summary
- ✅ 95% CI for mean sleep (t-interval)
- ✅ One-sample t-test vs μ₀ = 8h
- ✅ One-proportion z-test vs p₀ = 0.25
- ✅ Correlation r = Sxy/√(Sxx·Syy)
- ✅ Least-squares regression: b₁ = Sxy/Sxx, b₀ = ȳ − b₁x̄
- ✅ SE(b₁), t-statistic for slope, p-value, 95% CI for slope
- ✅ Two-sample Welch t-test (GPA by sleep group)
- ✅ Condition checks printed in plain English before each procedure

---

## References
- CDC (2023). *Sleep in Middle and High School Students*. cdc.gov/sleep
- College Board (2024). *AP Statistics Course and Exam Description*
- Hirshkowitz, M. et al. (2015). National Sleep Foundation sleep time recommendations. *Sleep Health*, 1(1), 40–43
- ASA (2025). *K–12 Statistics Education Project Competition Guidelines*
- Primary data: Anonymous Google Form survey, Doral Academy High, March 2026 (n = 99)
