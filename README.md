# Sleep & Academic Performance — AP Statistics Project

## Research Question
Is there a significant relationship between how many hours students sleep on school nights and their unweighted GPA? We also wanted to estimate the true mean sleep time for Doral Academy students and compare it against the CDC's recommendation of 8 hours.

---

## What We Found

| | |
|---|---|
| Mean sleep per night | 6.42 hours (95% CI: 6.20 – 6.64h) |
| vs. CDC benchmark (8h) | t = −14.48, p ≈ 0.000 — students sleep significantly less |
| Students sleeping ≥ 8h | Only 12.1%, vs. 25% nationally (p = 0.0015) |
| Sleep vs. GPA correlation | r = −0.011, p = 0.917 — no significant relationship |
| GPA by sleep group | p = 0.978 — no significant difference |

---

## Repo Structure

```
sleep-gpa-stats/
├── README.md
├── data/
│   └── Statistics_Research_.csv
├── analysis/
│   ├── sleep_gpa_analysis.py
│   └── requirements.txt
└── dashboard/
    ├── index.html
    ├── fig1_sleep_histogram.png
    ├── fig2_scatter_regression.png
    ├── fig3_gpa_boxplot_by_sleep.png
    ├── fig4_residuals.png
    └── results_summary.csv
```

---

## Running the Script

**Locally:**
```bash
pip install -r analysis/requirements.txt
cd analysis
python sleep_gpa_analysis.py
```

**On Replit:**  
Import from GitHub, upload the CSV to `data/`, set the run command to `python analysis/sleep_gpa_analysis.py`, and hit Run.

**Dashboard:**  
Enable GitHub Pages (Settings → Pages → source: main → /dashboard) and it'll be live at `https://YOUR_USERNAME.github.io/sleep-gpa-stats/`

---

## AP Units Covered

| Unit | Where |
|---|---|
| Unit 1 — One-Variable Data | Histograms, boxplots, five-number summary |
| Unit 2 — Two-Variable Data | Scatterplot, regression, correlation |
| Unit 3 — Collecting Data | Survey design, sampling method |
| Unit 6 — Inference for Proportions | One-proportion z-test |
| Unit 7 — Inference for Means | One-sample t-test, two-sample t-test |
| Unit 8 — Inference for Regression | Test for slope, CI for slope |

---

## About the Code

Every statistic in the script is calculated by hand first using the actual formula, then double-checked against scipy. This covers the rubric's extra credit requirement for annotated Python verification of all hand calculations. The sections are labeled clearly so the script can go straight into the appendix.
