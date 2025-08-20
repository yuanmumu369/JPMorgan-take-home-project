 Overview

This repo contains two Jupyter notebooks that walk from Exploratory Data Analysis (EDA) to a first leakage-safe baseline model for predicting bad_flag (loan default). The workflow is intentionally simple and reproducible: open each notebook and run all cells from top to bottom. Since there is no data set in the repo, please add the two orignal data set under data file.

notebooks/EDA.ipynb — cleans the target, parses key fields, visualizes data quality and signal, and surfaces potential data-leakage risks.

notebooks/modeling.ipynb — builds a leakage-aware baseline (train/valid split before any fitting, proper preprocessing, and class-imbalance handling), reports AUC and basic diagnostics.

How to Run


!!!!! Please create a new folder called data, and place the two csv files in there !!!!!!!!



Ensure Python 3.10+ and the following packages: pandas, numpy, matplotlib, scikit-learn. and correct requirments

Place the training CSV in the expected path or adjust the read path at the top of each notebook.

Open notebooks/EDA.ipynb and Run All cells, top → bottom.

Open notebooks/modeling.ipynb and Run All cells, top → bottom.


EDA structure:

My EDA is a step by step: 1, load the data  2, basic shape and schema  3, find the target balance and clean bad data 

4, find the missingness ratio  5, study numeric factor and categorical factor  6, I parse the string into data we can use  7, distribution pie chart to study the data imbalance  8, draw the deciles to study relationship 

9, draw a dot graph to study percent_bc_gt_75 VS revol_util_num and bad flag  10, study the top 6 correlated features

11, draw the correlation heat map  12, further study the data by having AUC and IV  13, check data hygiene


My ideas:

![distribution pie chart](figures/distribution_pie_chart.png)

Bad (1) is a tiny slice compared to Good (0). This isn’t a 50/50 problem—it’s a rare-event problem. If we chase raw accuracy, a dumb model that predicts ‘always Good’ will look great and still miss every Bad case, which is useless.

So I’ll (1) use stratified train/val splits so the bad rate stays consistent, (2) make the model care about the minority with class weights or oversampling, and (3) judge performance with ROC-AUC but put extra emphasis on PR-AUC and recall/precision at a threshold that matches our costs. I’ll also check calibration so predicted bad rates roughly match what the pie shows. Bottom line: success means surfacing the few Bad (1) cases without flooding ops with false alarms.


![deciles](figures/deciles.png)

The decile plot shows a clear, monotonic risk lift as total revolving utilization rises. At low utilization (≈10–30%), the bad rate sits around 5–6%. It wobbles slightly near 40% (a tiny dip likely from sampling noise), then climbs steadily: mid-50s to low-60s utilization yields ~6–7%, 70–80% moves to ~7.5–8%, and once utilization exceeds ~85–90% the bad rate jumps past ~9%. In other words, each higher utilization band carries meaningfully higher default risk, with an accelerating increase beyond ~60% and a notable step-up beyond ~85%.


![75VSrevol](figures/75VSrevol.png)


This chart relates percent_bc_gt_75 (share of bankcards above 75% utilization) to revol_util_num (overall revolving utilization), colored by bad_flag. Default risk rises monotonically along both axes: the lower-left region (percent_bc_gt_75 < 20% and revol_util_num < 40%) is dominated by goods, while the upper-right (percent_bc_gt_75 > 60% and revol_util_num > 70%) shows a visibly higher density of bads. This suggests that having many near-maxed cards and high total utilization is a strong risk signal. The vertical bands on the x-axis come from rounded integer values, but the pattern remains clear—more cards over 75% and higher overall utilization jointly correlate with a higher probability of bad_flag = 1. I also observe heaping at round percentages—50, 60, 65—where bad_flag cases look more concentrated, likely due to discretization/rounding rather than true risk jumps



![Top6](figures/Top6.png)

inq_last_6mths (recent hard pulls). Most applicants sit at 0–2 inquiries, and the Bad (1) bars get noticeably thicker around 2–3. That’s classic “credit seeking” risk—people shopping for/depending on new credit. I’d keep this feature, cap it at 3 (0,1,2,≥3) or bin it, so the tail doesn’t add noise while still preserving the lift.

mths_since_recent_inq (months since last pull). The histogram is sharply left-skewed and bads concentrate at very small values (0–3 months). This is the mirror image of the first feature: a very recent pull aligns with higher risk. Treat it as monotonic (smaller → riskier); an inverse or binned version (e.g., 0,1–3,4–6,>6) will be stable.

percent_bc_gt_75 (share of bankcards over 75% utilized). You see spikes at neat values (50/75/100) because of bureau rounding, and the red overlay thickens as the share increases—especially near 100%. That’s a strong stress signal: many cards close to the limit. I’d either keep it numeric with capping [0,100] or use intuitive bins (0, (0,25], (25,50], (50,75], (75,100]) to capture the step-up in risk.

total_bc_limit (total bankcard limit). Heavy right tail; bads cluster in the low-limit region. Lower capacity often means thinner/younger files or weaker credit. Log-transform (log1p) or winsorize to tame the tail; expect a negative relationship to risk after transformation (higher limits → lower bad rate, all else equal).

tot_hi_cred_lim (total highest credit limit). Same story as total_bc_limit—long tail and more bads at the low end—so it corroborates the “capacity matters” theme. It may overlap strongly with total_bc_limit; I’d keep both for now but watch multicollinearity, letting regularization decide or later consolidating into a single “capacity index.”

bc_util (overall bankcard utilization %). As utilization rises, the bad overlay thickens; your bad-rate-by-decile plot shows a clear climb from ~60% and a sharp step above ~80–90%. This is a powerful, monotonic predictor. Cap at 100 and consider nonlinear treatment (knots/bins around 60/80/90). Also worth interacting with percent_bc_gt_75 to capture the “many cards high & overall high” risk pocket.



![Heatmap](figures/heatMap.png)


What the heatmap tells us. Two feature pairs are nearly duplicates: total_bc_limit vs tot_hi_cred_lim (≈1.00) and percent_bc_gt_75 vs bc_util (≈0.83). That’s classic multicollinearity—keep one from each pair (or regularize) so the model doesn’t waste capacity on redundant signals. Inquiry recency behaves as expected: inq_last_6mths is strongly negative with mths_since_recent_inq (≈−0.68), i.e., more recent pulls → fewer months since last pull. Capacity features are moderately negative with utilization (≈−0.25 to −0.29): higher limits generally mean lower revolving usage. Finally, the direct linear correlations with bad_flag are small (~0.04–0.06). That tells us the target signal is likely non-linear and interaction-driven



![AUC](figures/auc.png)
![IV](figures/IV.png)


The two ranking exhibits convey a consistent narrative. Interest rate (int_rate/int_rate_num) dominates both the signed univariate AUC and the Information Value, which strongly suggests embedded risk-pricing and potential target leakage; for a credit-risk model we should exclude rate variables. Once rate is removed, the principal sources of signal are the utilization family (revol_util_num, bc_util, percent_bc_gt_75): their positive signed AUCs and non-trivial IVs align with our earlier scatter and decile plots showing monotonic increases in bad rate as utilization rises and as a larger share of bankcards exceed 75% usage. Capacity/limit measures (tot_hi_cred_lim, total_bc_limit, tot_cur_bal) exhibit protective effects (negative signed AUC) but are highly collinear, so a single representative or regularization is advisable. Inquiry timing and intensity behave as expected—more/closer inquiries correspond to higher risk—while purpose and home_ownership contribute modest, secondary discrimination. Overall, aside from rate, most IVs fall below 0.1, indicating weak-to-moderate univariate power; therefore, performance will rely on combining features rather than any single predictor. 



![Modeling results](figures/modeling.png)

This is my modeling results
