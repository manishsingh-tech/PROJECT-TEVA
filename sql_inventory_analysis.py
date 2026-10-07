import sqlite3
import pandas as pd
import numpy as np

# 1. Load the Pharma sales dataset
df = pd.read_csv("pharma_sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Connect to SQLite database
conn = sqlite3.connect("pharma_inventory.db")
df.to_sql("sales_data", conn, if_exists="replace", index=False)

print("--- 1. SQL CTE & WINDOW FUNCTIONS: MoM Growth & Stock Availability ---")

# 2. Advanced SQL Query with CTEs and Window Functions (LAG)
query_mom = """
WITH MonthlySales AS (
    SELECT 
        SKU_ID,
        Region,
        strftime('%Y-%m', Date) AS Sales_Month,
        SUM(Units_Sold) AS Total_Units_Sold,
        AVG(Stock_Level) AS Avg_Stock_Level
    FROM sales_data
    GROUP BY SKU_ID, Region, Sales_Month
),
MoM_Calculations AS (
    SELECT 
        SKU_ID,
        Region,
        Sales_Month,
        Total_Units_Sold,
        Avg_Stock_Level,
        LAG(Total_Units_Sold, 1) OVER (PARTITION BY SKU_ID, Region ORDER BY Sales_Month) AS Prev_Month_Sales
    FROM MonthlySales
)
SELECT 
    SKU_ID,
    Region,
    Sales_Month,
    Total_Units_Sold,
    Prev_Month_Sales,
    ROUND(
        ((Total_Units_Sold - Prev_Month_Sales) * 100.0 / NULLIF(Prev_Month_Sales, 0)), 2
    ) AS MoM_Growth_Pct,
    CASE 
        WHEN Avg_Stock_Level < (Total_Units_Sold * 1.2) THEN '⚠️ High Stockout Risk'
        ELSE '✅ Healthy Stock'
    END AS Stockout_Risk_Status
FROM MoM_Calculations;
"""

df_mom_result = pd.read_sql(query_mom, conn)
print(df_mom_result.head(10))

print("\n--- 2. ECONOMIC ORDER QUANTITY (EOQ) & REORDER ANALYSIS ---")

query_eoq = """
SELECT 
    SKU_ID,
    SUM(Units_Sold) AS Annual_Demand,
    AVG(Unit_Price_INR) AS Avg_Unit_Price
FROM sales_data
GROUP BY SKU_ID
"""
df_eoq = pd.read_sql(query_eoq, conn)

ORDERING_COST = 2500.0
df_eoq['Holding_Cost_Per_Unit'] = df_eoq['Avg_Unit_Price'] * 0.15
df_eoq['EOQ_Units'] = np.sqrt((2 * df_eoq['Annual_Demand'] * ORDERING_COST) / df_eoq['Holding_Cost_Per_Unit'])
df_eoq['EOQ_Units'] = np.ceil(df_eoq['EOQ_Units'])

print(df_eoq[['SKU_ID', 'Annual_Demand', 'Avg_Unit_Price', 'EOQ_Units']])
conn.close()