import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import os
import glob

# --- Hide Streamlit default UI elements ---
hide_streamlit_style = """
    <style>
    /* Hide deploy button, hamburger menu, footer, and top bar */
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {visibility: hidden !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stAppDeployButton {visibility: hidden !important;}
    /* Optional: hide rainbow top progress bar */
    div[data-testid="stProgressBar"] {visibility: hidden !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'csv_data'))
LABELS = ["H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A", "W", "1", "2", "3", "4", "R"]

# Collect all CSV files in DATA_DIR
csv_paths = glob.glob(os.path.join(DATA_DIR, '*.csv'))
dfs = []
for p in csv_paths:
    try:
        df = pd.read_csv(p)
        dfs.append(df)
    except Exception:
        st.warning(f"Could not read CSV: {p}")

if len(dfs) == 0:
    st.warning(f"No CSV files found in {DATA_DIR}. Place extracted feature CSVs there.")
else:
    df_all = pd.concat(dfs, ignore_index=True).fillna(0)

    st.title("Statistical Analysis of Extracted Dataset")

    # 1. Dataset Shape
    st.markdown(f"**Dataset Shape:** {df_all.shape[0]} samples, {df_all.shape[1]} columns")

    # 2. Descriptive Stats
    st.subheader("Descriptive Statistics")
    st.dataframe(df_all.describe().transpose())

    # 3. Label Distribution (if labels exist)
    if set(LABELS).issubset(set(df_all.columns)):
        st.subheader("Label Distribution")
        label_counts = df_all[LABELS].sum().sort_values(ascending=False)
        st.bar_chart(label_counts)

    # 5. Quick correlation heatmap for numeric features
    num = df_all.select_dtypes(include=['number'])
    if num.shape[1] > 1:
        st.subheader("Correlation Matrix (numeric features)")
        corr = num.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, ax=ax, cmap='coolwarm')
        st.pyplot(fig)
