import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Page Config
st.set_page_config(
    page_title="Retail Sales Intelligence Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Dark Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    section[data-testid="stSidebar"] { background-color: #1e2130; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='color:#00d4ff;'>🛒 Retail Sales Intelligence & Forecasting System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:gray;'>Powered by Python | SQLite | Streamlit | Machine Learning</p>", unsafe_allow_html=True)
st.divider()

# Load Data
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "retail_sales_powerbi.csv")
df = pd.read_csv(csv_path)
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

# Sidebar Filters
st.sidebar.markdown("## 🎛️ Filters")
region = st.sidebar.selectbox("Select Region", ["All"] + sorted(df["Region"].unique().tolist()))
category = st.sidebar.selectbox("Select Category", ["All"] + sorted(df["Category"].unique().tolist()))

# Apply Filters
filtered = df.copy()
if region != "All":
    filtered = filtered[filtered["Region"] == region]
if category != "All":
    filtered = filtered[filtered["Category"] == category]

# KPIs
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Sales", f"₹{filtered['Sales'].sum():,.0f}")
col2.metric("📈 Total Profit", f"₹{filtered['Profit'].sum():,.0f}")
col3.metric("🛍️ Total Orders", f"{len(filtered):,}")
col4.metric("📦 Avg Order Value", f"₹{filtered['Sales'].mean():,.0f}")

st.divider()

# Dataset Preview
st.markdown("### 📄 Dataset Preview")
st.dataframe(filtered.head(5), use_container_width=True)

st.divider()

# Row 1 - Bar + Pie
st.markdown("### 📊 Sales & Profit Analysis")
col1, col2 = st.columns(2)

with col1:
    sales_cat = filtered.groupby("Category")["Sales"].sum().reset_index()
    fig1 = px.bar(
        sales_cat, x="Category", y="Sales",
        color="Category", title="Sales by Category",
        template="plotly_dark",
        color_discrete_sequence=["#00d4ff", "#ff6b6b", "#ffd93d"]
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    profit_cat = filtered.groupby("Category")["Profit"].sum().reset_index()
    fig2 = px.pie(
        profit_cat, names="Category", values="Profit",
        title="Profit by Category",
        template="plotly_dark",
        color_discrete_sequence=["#00d4ff", "#ff6b6b", "#ffd93d"]
    )
    st.plotly_chart(fig2, use_container_width=True)

# Row 2 - Top 10 + Scatter
col3, col4 = st.columns(2)

with col3:
    top10 = filtered.groupby("Sub_Category")["Sales"].sum().nlargest(10).reset_index()
    fig3 = px.bar(
        top10, x="Sales", y="Sub_Category",
        orientation="h", title="Top 10 Sub-Categories by Sales",
        template="plotly_dark",
        color="Sales", color_continuous_scale="Blues"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.scatter(
        filtered, x="Discount", y="Sales",
        color="Category", title="Discount vs Sales Impact",
        template="plotly_dark",
        color_discrete_sequence=["#00d4ff", "#ff6b6b", "#ffd93d"]
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# Sales Trend
st.markdown("### 📉 Sales Trend Over Time")
sales_trend = filtered.groupby("Order_Date")["Sales"].sum().reset_index()
fig5 = px.line(
    sales_trend, x="Order_Date", y="Sales",
    title="Monthly Sales Trend",
    template="plotly_dark",
    color_discrete_sequence=["#00d4ff"]
)
st.plotly_chart(fig5, use_container_width=True)

st.divider()

# Forecasting
st.markdown("### 🔮 Sales Forecasting (Next 30 Days)")
try:
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import plotly.graph_objects as go

    trend = filtered.groupby("Order_Date")["Sales"].sum().reset_index()
    trend = trend.dropna()
    trend["days"] = (trend["Order_Date"] - trend["Order_Date"].min()).dt.days

    X = trend["days"].values.reshape(-1, 1)
    y = trend["Sales"].values

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(
        trend["days"].max() + 1,
        trend["days"].max() + 31
    ).reshape(-1, 1)

    future_sales = model.predict(future_days)
    future_dates = pd.date_range(
        trend["Order_Date"].max(), periods=30, freq="D"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Sales": future_sales
    })

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=trend["Order_Date"], y=trend["Sales"],
        name="Actual Sales",
        line=dict(color="#00d4ff", width=2)
    ))
    fig6.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Forecasted Sales"],
        name="Forecasted Sales",
        line=dict(color="#ff6b6b", width=2, dash="dash")
    ))
    fig6.update_layout(
        template="plotly_dark",
        title="Actual vs Forecasted Sales"
    )
    st.plotly_chart(fig6, use_container_width=True)

except Exception as e:
    st.warning(f"Forecasting error: {e}")

st.divider()
st.success("✅ Dashboard loaded successfully! | Retail Sales Intelligence System")