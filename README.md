#  Pharma Supply Chain Demand Forecasting & Inventory Optimizer

An enterprise-grade data science application designed to predict regional pharmaceutical demand, prevent stockouts, and optimize safety stock thresholds using machine learning.

---

##  Project Overview
Pharmaceutical supply chains require highly accurate demand forecasting to prevent critical drug shortages (stockouts) while avoiding capital blockage from overstocking. This project implements an end-to-end Machine Learning pipeline and an interactive Streamlit web dashboard to analyze historical sales, track seasonal trends, and automate reorder alerts across multiple regional hubs and Product SKUs.

##  Tech Stack
* **Language:** Python 3.14.7
* **Core Libraries:** Pandas, NumPy
* **Machine Learning:** XGBoost Regressor, Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Interactive UI:** Streamlit
* **Environment:** VS Code, Jupyter Notebook

---

##  Key Features & Methodology
1. **Synthetic Data Engineering:** Generated a robust dataset (`720+ rows`) simulating 3 years of multi-regional sales, inventory levels, lead times, and unit pricing with embedded winter seasonal multipliers.
2. **Time-Series Feature Engineering:** 
   - **Lag Features (`Lag_1_Units_Sold`):** Captures immediate preceding monthly demand patterns.
   - **Rolling Averages (`Rolling_Avg_3M`):** Smooths out volatility to reflect stable multi-month sales trends.
3. **Machine Learning Model (XGBoost):**
   - Trained an **XGBoost Regressor** on 80-20 train-test split.
   - Achieved an **R² Score of ~0.8947** and an **RMSE of 477.84**.
   - **Feature Importance Analysis:** Identified `Rolling_Avg_3M` as the dominant predictor (~94.3% importance).
4. **Supply Chain Optimization Engine:**
   - Calculates **Optimal Safety Stock** with a 20% buffer over predicted demand.
   - Automatically generates real-time **Reorder Alerts** when current stock dips below required safety thresholds.
5. **Interactive Dashboard:** Deployed as a web application using Streamlit for dynamic filtering by SKU and Region.

---

##  Installation & Running Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/pharma-supply-chain-optimizer.git](https://github.com/YOUR_USERNAME/pharma-supply-chain-optimizer.git)
   cd pharma-supply-chain-optimizer
