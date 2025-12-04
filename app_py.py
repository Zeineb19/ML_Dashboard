import streamlit as st
import pandas as pd
import plotly.express as px
import os

# -----------------------------
#    FX VOLATILITY DASHBOARD
# -----------------------------

st.title("FX Volatility Dashboard")
st.write("Select an exchange rate to visualize its historical trend.")

# 1️⃣ List all available CSV files inside exchange_rate_results
data_folder = "exchange_rate_results"

files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

if not files:
    st.error("❌ No CSV files found in exchange_rate_results folder.")
else:
    selected_file = st.selectbox("Choose an exchange rate:", files)

    #  Load selected dataset
    df = pd.read_csv(os.path.join(data_folder, selected_file))

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Expected structure: date / close
    if "date" not in df.columns or "close" not in df.columns:
        st.error(f"❌ File {selected_file} must contain 'Date' and 'Close' columns.")
    else:
        # Convert date column
        df["date"] = pd.to_datetime(df["date"])

        #  Plot
        fig = px.line(
            df,
            x="date",
            y="close",
            title=f"📈 {selected_file.replace('.csv','').upper()} Exchange Rate",
        )

        st.plotly_chart(fig, use_container_width=True)

        #  Display dataframe
        with st.expander("📄 Show raw data"):
            st.dataframe(df)
