import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =========================================
# CONFIG
# =========================================
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'csv_data'))
TARGET_FS = 100  # consistent resample frequency

st.set_page_config(page_title="Signal Analysis", layout="wide")
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

st.sidebar.title("ECG / HRV Feature Explorer")
csv_files = [os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, '*.csv'))]
selected_file = st.sidebar.selectbox("Choose a features CSV", csv_files)

st.title("ECG / HRV Feature Explorer")
st.markdown("Explore extracted HR/HRV and related physiological features. Select visualizations below.")


def numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def candidate_hrv_cols(cols):
    keywords = ['hr', 'rr', 'sdnn', 'rmssd', 'pnn50', 'lf_hf', 'lf', 'hf']
    return [c for c in cols if any(k in c.lower() for k in keywords)]


# =========================================
# LOAD FEATURE CSV
# =========================================
if not selected_file:
    st.info("No CSV files found in web/csv_data. Add feature CSVs and reload the app.")
    st.stop()

path = os.path.join(DATA_DIR, selected_file)
try:
    df = pd.read_csv(path)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader(f"Preview — {selected_file}")
st.dataframe(df.head())

num_cols = numeric_columns(df)
hrv_cols = candidate_hrv_cols(df.columns)

# -----------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------
st.sidebar.subheader("Visualizations")
viz_options = {
    'Histogram': st.sidebar.checkbox('Histogram', True),
    'Boxplot': st.sidebar.checkbox('Boxplot', True),
    'Violin': st.sidebar.checkbox('Violin', False),
    'Time-series': st.sidebar.checkbox('Time-series (row index)', False),
    'Rolling mean': st.sidebar.checkbox('Rolling mean (windowed)', False),
    'Correlation heatmap': st.sidebar.checkbox('Correlation heatmap', True),
    'PCA': st.sidebar.checkbox('PCA (2D scatter + variance)', False),
    't-SNE': st.sidebar.checkbox('t-SNE (2D)', False),
    'Pairplot (slow)': st.sidebar.checkbox('Pairplot (slow)', False),
}

label_cols = [c for c in df.columns if df[c].isin(['low','high',0,1]).any() and df[c].nunique() < 20]
single_feat = st.sidebar.selectbox('Feature for single-column plots', num_cols, index=0) if len(num_cols) > 0 else None

# -----------------------------------------
# BASIC VISUALIZATIONS
# -----------------------------------------
if viz_options['Histogram'] and single_feat:
    st.subheader(f"Histogram — {single_feat}")
    fig, ax = plt.subplots(figsize=(10, 4))
    _ = ax.hist(df[single_feat].dropna(), bins=30, color='teal', edgecolor='black', alpha=0.8)
    _ = ax.set_xlabel(single_feat)
    _ = ax.set_ylabel('Count')
    st.pyplot(fig)

grouping = st.sidebar.selectbox('Optional grouping column', [None] + label_cols) if len(label_cols) > 0 else None

if viz_options['Boxplot'] and single_feat:
    st.subheader(f"Boxplot — {single_feat}")
    fig, ax = plt.subplots(figsize=(10, 4))
    _ = sns.boxplot(x=grouping, y=single_feat, data=df, ax=ax) if grouping else sns.boxplot(y=df[single_feat].dropna(), ax=ax)
    st.pyplot(fig)

if viz_options['Violin'] and single_feat:
    st.subheader(f"Violin — {single_feat}")
    fig, ax = plt.subplots(figsize=(10, 4))
    _ = sns.violinplot(x=grouping, y=single_feat, data=df, ax=ax) if grouping else sns.violinplot(y=df[single_feat].dropna(), ax=ax)
    st.pyplot(fig)

# Correlation heatmap
if viz_options['Correlation heatmap'] and len(num_cols) > 1:
    st.subheader('Correlation matrix (numeric features)')
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    _ = sns.heatmap(corr, ax=ax, cmap='coolwarm')
    st.pyplot(fig)

# PCA
if viz_options['PCA'] and len(num_cols) > 1:
    st.subheader('PCA (first two components)')
    pca_cols = st.sidebar.multiselect('Select columns for PCA', num_cols, default=num_cols[:min(10,len(num_cols))])
    if len(pca_cols) >= 2:
        X = df[pca_cols].fillna(0).to_numpy()
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(8, 6))
        _ = sns.scatterplot(x=Xp[:,0], y=Xp[:,1], hue=df[grouping], ax=ax, palette='tab10') if grouping else ax.scatter(Xp[:,0], Xp[:,1], s=20)
        _ = ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        _ = ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        _ = ax.set_title('PCA projection')
        st.pyplot(fig)

# =========================================
# RAW SIGNAL VISUALIZATION & R-PEAKS
# =========================================
st.markdown('---')
st.subheader('Raw Signal Visualization with R-peak Detection')

record_name = selected_file.replace('_features.csv', '')
dataset_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset'))
record_path = os.path.join(dataset_dir, record_name)

run_raw = st.checkbox('Show raw signal with R-peak detection (requires wfdb & biosppy)', False)
window_size = st.slider('Window size (samples)', 1000, 10000, 5000)

if run_raw:
    try:
        import wfdb
        from scipy.signal import butter, filtfilt, savgol_filter, resample
        from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks

        rec = wfdb.rdrecord(record_path)
        available_signals = rec.sig_name
        selected_signal = st.selectbox('Select signal to visualize', available_signals)
        signal_idx = available_signals.index(selected_signal)

        sig = rec.p_signal[:, signal_idx]
        fs = int(rec.fs) if hasattr(rec, 'fs') else 100

        # 🔁 Resample for consistency
        if fs != TARGET_FS:
            duration = len(sig) / fs
            new_len = int(duration * TARGET_FS)
            sig = resample(sig, new_len)
            st.info(f"Resampled from {fs} Hz → {TARGET_FS} Hz ({len(sig)} samples)")
            fs = TARGET_FS

        # ========== PLOTS ==========
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw Signal**")
            fig1, ax1 = plt.subplots(figsize=(12, 4))
            ax1.plot(sig[:window_size])
            ax1.set_title(f'Raw {selected_signal} Signal')
            ax1.set_xlabel('Samples')
            ax1.set_ylabel('Amplitude')
            st.pyplot(fig1)

        with col2:
            st.markdown("**Filtered Signal with R-peaks**")

            nyq = fs / 2
            low_freq = 0.5
            high_freq = min(40.0, nyq * 0.95)

            wn_low = max(1e-5, low_freq / nyq)
            wn_high = min(0.999, high_freq / nyq)

            if wn_high <= wn_low or wn_high >= 1.0:
                st.warning(f"⚠️ Invalid band limits for fs={fs} Hz. Using raw signal.")
                filtered = sig.copy()
            else:
                b, a = butter(2, [wn_low, wn_high], btype='bandpass')
                filtered = filtfilt(b, a, sig)

            # Optional smoothing
            window = max(5, int(fs * 0.1))
            if window > 3:
                filtered = savgol_filter(filtered, window if window % 2 == 1 else window + 1, 3)

            filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)

            if 'ecg' in selected_signal.lower():
                try:
                    rpeaks, = hamilton_segmenter(filtered, sampling_rate=fs)
                    rpeaks, = correct_rpeaks(filtered, rpeaks, sampling_rate=fs, tol=0.1)
                    rr = rpeaks[rpeaks < window_size]
                    fig2, ax2 = plt.subplots(figsize=(12, 4))
                    ax2.plot(filtered[:window_size], label='Filtered')
                    if len(rr) > 0:
                        ax2.scatter(rr, filtered[rr], color='red', s=30, label=f'R-peaks ({len(rr)})')
                        rr_intervals = np.diff(rr) / fs
                        if len(rr_intervals) > 0:
                            hr = 60 / np.mean(rr_intervals)
                            st.markdown(f"""
                            **Quick HRV Stats:**
                            - Mean Heart Rate: {hr:.1f} BPM
                            - Mean RR: {np.mean(rr_intervals)*1000:.0f} ms
                            - SDNN: {np.std(rr_intervals)*1000:.0f} ms
                            """)
                    else:
                        st.warning("No R-peaks detected in this window.")
                    ax2.set_title('Filtered ECG with R-peaks')
                    ax2.legend()
                    st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Error detecting R-peaks: {e}")
            else:
                st.info("Non-ECG signal — showing filtered waveform only.")
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.plot(filtered[:window_size])
                ax2.set_title(f'Filtered {selected_signal}')
                st.pyplot(fig2)

        # Record info
        st.markdown("**Record Info:**")
        st.write(f"- Sampling rate: {fs} Hz")
        st.write(f"- Available signals: {', '.join(available_signals)}")
        st.write(f"- Duration: {len(sig)/fs:.1f} s")

    except ModuleNotFoundError:
        st.warning("Install wfdb and biosppy to use raw visualization: `pip install wfdb biosppy`")
    except FileNotFoundError:
        st.error(f"Could not find raw signal file: {record_path}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
