import streamlit as st
import pandas as pd
import datetime as dt
import subprocess
import os

# --- CONFIG ---
BATCH_CSV = "batch_signal_scores.csv"
LOG_FILE = "batchrun.log"  # Set to your logger file

st.set_page_config(page_title="QuantBot Signal Dashboard", layout="wide")

st.title("📈 QuantBot Signal Dashboard")

# --- SIDEBAR ---
st.sidebar.header("Controls")

# Load Data
@st.cache_data
def load_batch_data():
    if not os.path.exists(BATCH_CSV):
        return pd.DataFrame()
    df = pd.read_csv(BATCH_CSV)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

data = load_batch_data()

# --- Symbol Filter ---
symbols = sorted(data["symbol"].unique()) if not data.empty else []
selected_symbols = st.sidebar.multiselect(
    "Select Symbols", options=symbols, default=symbols or ["AAPL"]
)

# --- Date Selector ---
if not data.empty and "date" in data.columns:
    dates = pd.to_datetime(data["date"]).dt.date.unique()
    date_selected = st.sidebar.selectbox(
        "Select Date", options=sorted(dates, reverse=True), index=0
    )
else:
    date_selected = dt.date.today()

# --- Run Batch Button ---
if st.sidebar.button("Run New Batch Now"):
    with st.spinner("Running batch_signal_test.py..."):
        result = subprocess.run(["python3", "batch_signal_test.py"], capture_output=True, text=True)
        st.sidebar.success("Batch completed.")
        st.sidebar.write(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        st.cache_data.clear()  # force reload
    data = load_batch_data()  # Reload after run

# --- Filter data for UI ---
if not data.empty:
    data = data[data["symbol"].isin(selected_symbols)]
    data = data[data["date"].dt.date == date_selected]

# --- MAIN TABLE ---
st.subheader("Batch Signal Scores")

if data.empty:
    st.warning("No batch results found for the selected filter.")
else:
    st.dataframe(
        data[["symbol", "score", "sentiment_source", "n_headlines", "status", "error_message"]].set_index("symbol"),
        use_container_width=True,
        hide_index=False
    )

    # --- Expandable Rows for Headlines ---
    st.subheader("Headlines per Symbol")
    for i, row in data.iterrows():
        with st.expander(f"{row['symbol']} - Score: {row['score']}"):
            st.write(f"**Sentiment Source:** {row.get('sentiment_source','')}")
            # Optionally, display top headlines if you add a field with joined headlines
            # st.write(row.get("headlines", ""))
            st.write(f"**Error/Status:** {row['status']} {row['error_message']}")

    # --- Signal Score Chart Over Time ---
    if "date" in data.columns and "score" in data.columns:
        chart_df = load_batch_data()
        chart_df = chart_df[chart_df["symbol"].isin(selected_symbols)]
        chart_df = chart_df.sort_values(by=["symbol", "date"])
        st.subheader("Signal Score Over Time")
        st.line_chart(
            data=chart_df.pivot(index="date", columns="symbol", values="score"),
            use_container_width=True,
        )

# --- LOG PANE (Optional) ---
if st.checkbox("Show Batch Run Log", value=False):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            log_text = f.readlines()[-100:]  # last 100 lines
        st.code("".join(log_text), language="log")
    else:
        st.warning("No log file found.")

st.caption("© QuantBot Research - Powered by Streamlit")
