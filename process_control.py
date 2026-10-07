import pandas as pd
import numpy as np
from scipy import stats

# 1. Simulate Manufacturing Batch Quality Data
np.random.seed(42)
n_records = 500

data = {
    'Batch_ID': [f'BATCH_{i:03d}' for i in range(1, n_records + 1)],
    'Shift': np.random.choice(['Morning', 'Evening', 'Night'], size=n_records),
    'Supplier': np.random.choice(['Vendor_A', 'Vendor_B', 'Vendor_C'], size=n_records),
    # Active Pharmaceutical Ingredient (API) Purity percentage
    'API_Purity_Pct': np.random.normal(loc=98.5, scale=0.8, size=n_records),
    'Defect_Flag': np.random.choice([0, 1], size=n_records, p=[0.92, 0.08])
}

df_qc = pd.DataFrame(data)

# Introduce outliers to simulate process failures
df_qc.loc[10, 'API_Purity_Pct'] = 94.2
df_qc.loc[45, 'API_Purity_Pct'] = 101.5

print("--- RAW BATCH QUALITY DATA (SAMPLE) ---")
print(df_qc.head())

# 2. Statistical Process Control (SPC) - Control Limits & Z-scores
mean_purity = df_qc['API_Purity_Pct'].mean()
std_purity = df_qc['API_Purity_Pct'].std()

# 3-Sigma Control Limits
ucl = mean_purity + (3 * std_purity)
lcl = mean_purity - (3 * std_purity)

df_qc['Z_Score'] = (df_qc['API_Purity_Pct'] - mean_purity) / std_purity
df_qc['SPC_Outlier'] = (df_qc['API_Purity_Pct'] > ucl) | (df_qc['API_Purity_Pct'] < lcl)

print(f"\n--- SPC CONTROL LIMITS ---")
print(f"Process Mean: {mean_purity:.2f} | UCL (+3σ): {ucl:.2f} | LCL (-3σ): {lcl:.2f}")
print(f"Outliers Detected via SPC: {df_qc['SPC_Outlier'].sum()} batches")

# 3. Hypothesis Testing using SciPy
print(f"\n--- HYPOTHESIS TESTING ---")

morning_purity = df_qc[df_qc['Shift'] == 'Morning']['API_Purity_Pct']
evening_purity = df_qc[df_qc['Shift'] == 'Evening']['API_Purity_Pct']
night_purity = df_qc[df_qc['Shift'] == 'Night']['API_Purity_Pct']

anova_f_stat, anova_p_val = stats.f_oneway(morning_purity, evening_purity, night_purity)
print(f"1. ANOVA Test (Shifts vs Purity): F-Statistic = {anova_f_stat:.4f}, p-value = {anova_p_val:.4f}")

contingency_table = pd.crosstab(df_qc['Supplier'], df_qc['Defect_Flag'])
chi2_stat, chi2_p_val, dof, ex = stats.chi2_contingency(contingency_table)
print(f"\n2. Chi-Square Test (Supplier vs Defects): Chi2 = {chi2_stat:.4f}, p-value = {chi2_p_val:.4f}")