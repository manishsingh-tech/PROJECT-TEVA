import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Pharma Supply Chain & Demand Forecasting", 
    page_icon="💊", 
    layout="wide"
)

st.title("💊 Pharma Supply Chain Demand Forecasting & Inventory Optimizer")
st.markdown("An enterprise-grade data science application designed to predict pharmaceutical demand, prevent stockouts, and optimize regional inventory levels.")

@st.cache_data
def load_and_train_model():
    # Load dataset
    df = pd.read_csv("pharma_sales_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Feature Engineering
    df = df.sort_values(by=['SKU_ID', 'Region', 'Date']).reset_index(drop=True)
    df['Lag_1_Units_Sold'] = df.groupby(['SKU_ID', 'Region'])['Units_Sold'].shift(1)
    df['Rolling_Avg_3M'] = df.groupby(['SKU_ID', 'Region'])['Units_Sold'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    
    df['Lag_1_Units_Sold'] = df['Lag_1_Units_Sold'].fillna(df['Units_Sold'])
    df['Rolling_Avg_3M'] = df['Rolling_Avg_3M'].fillna(df['Units_Sold'])
    
    # Train XGBoost Model
    features = ['Lag_1_Units_Sold', 'Rolling_Avg_3M', 'Lead_Time_Days', 'Unit_Price_INR']
    target = 'Units_Sold'
    
    X = df[features]
    y = df[target]
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    df['Predicted_Demand'] = model.predict(X)
    df['Optimal_Safety_Stock'] = np.ceil(df['Predicted_Demand'] * 1.2)
    df['Reorder_Alert'] = df['Stock_Level'] < df['Optimal_Safety_Stock']
    
    return df, model, features

df, model, features = load_and_load_model() if 'load_and_load_model' in globals() else load_and_train_model()

# Sidebar Filters
st.sidebar.header("Supply Chain Parameters")
selected_sku = st.sidebar.selectbox("Select SKU ID", df['SKU_ID'].unique())
selected_region = st.sidebar.selectbox("Select Regional Hub", df['Region'].unique())

# Filter data based on user selection
filtered_df = df[(df['SKU_ID'] == selected_sku) & (df['Region'] == selected_region)]

st.subheader(f"📊 Inventory & Demand Dashboard: {selected_sku} ({selected_region})")

# Key Metrics Display
col1, col2, col3 = st.columns(3)
latest_row = filtered_df.iloc[-1] if not filtered_df.empty else None

if latest_row is not None:
    col1.metric("Current Stock Level", f"{int(latest_row['Stock_Level'])} Units")
    col2.metric("Predicted Demand", f"{int(latest_row['Predicted_Demand'])} Units")
    col3.metric("Optimal Safety Stock", f"{int(latest_row['Optimal_Safety_Stock'])} Units")
    
    if latest_row['Reorder_Alert']:
        st.error("⚠️ **REORDER ALERT**: Stock level is below the optimal safety stock threshold! Immediate replenishment required.")
    else:
        st.success("✅ **Stock Status Healthy**: Current stock levels are sufficient to meet forecasted demand.")

# Historical Chart
st.markdown("### 📈 Historical Sales vs Predicted Demand Trend")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(filtered_df['Date'], filtered_df['Units_Sold'], label='Actual Sales', marker='o', color='teal', linewidth=2)
ax.plot(filtered_df['Date'], filtered_df['Predicted_Demand'], label='Predicted Demand (XGBoost)', linestyle='--', marker='x', color='coral', linewidth=2)
ax.set_xlabel('Timeline (Monthly)', fontsize=11)
ax.set_ylabel('Units', fontsize=11)
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)
st.pyplot(fig)

# Raw Filtered Data Table
st.markdown("### 📋 Detailed Records View")
st.dataframe(filtered_df[['Date', 'SKU_ID', 'Product_Name', 'Region', 'Units_Sold', 'Stock_Level', 'Optimal_Safety_Stock', 'Reorder_Alert']], use_container_width=True)