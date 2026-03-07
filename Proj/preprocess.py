import os
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as signal
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = "dataset"
FEATURES_DIR = "features"
ANNOTATIONS_DIR = "annotations"
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

WINDOW_SIZE = 300  # 5 minutes in seconds
RESAMPLE_FS = 100  # use higher resampling for ECG/EMG accuracy

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------

def bandpass_filter(sig, fs, low, high):
    nyq = 0.5 * fs
    b, a = signal.butter(4, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, sig)

def lowpass_filter(sig, fs, cutoff):
    nyq = 0.5 * fs
    b, a = signal.butter(4, cutoff / nyq, btype="low")
    return signal.filtfilt(b, a, sig)

def resample_signal(sig, orig_fs, target_fs=RESAMPLE_FS):
    if orig_fs == target_fs:
        return sig
    return signal.resample(sig, int(len(sig) * target_fs / orig_fs))

# ------------------------------
# FEATURE EXTRACTION
# ------------------------------

# --- ECG / HRV ---
def extract_ecg_features(ecg, fs):
    # Bandpass ECG
    ecg_filt = bandpass_filter(ecg, fs, 0.5, 40)
    # R-peak detection
    diff = np.diff(ecg_filt)
    squared = diff**2
    win = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(win)/win, mode="same")
    peaks, _ = signal.find_peaks(integrated, distance=0.6*fs, height=np.mean(integrated))
    rr = np.diff(peaks) / fs  # in seconds
    
    features = {}
    if len(rr) > 2:
        features["HR_mean"] = 60.0 / np.mean(rr)
        features["SDNN"] = np.std(rr) * 1000
        diff_rr = np.diff(rr)
        features["RMSSD"] = np.sqrt(np.mean(diff_rr**2)) * 1000
        features["pNN50"] = np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr)
        f, psd = signal.welch(rr - np.mean(rr), fs=1.0/np.mean(rr), nperseg=min(256, len(rr)))
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        lf_power = np.trapz(psd[(f>=lf_band[0]) & (f<hf_band[1])], f[(f>=lf_band[0]) & (f<hf_band[1])])
        hf_power = np.trapz(psd[(f>=hf_band[0]) & (f<hf_band[1])], f[(f>=hf_band[0]) & (f<hf_band[1])])
        features["LF_HF"] = lf_power / hf_power if hf_power > 0 else 0.0
    else:
        features.update({"HR_mean":0, "SDNN":0, "RMSSD":0, "pNN50":0, "LF_HF":0})
    
    # RR intervals stats
    features["RR_mean"] = np.mean(rr) if len(rr)>0 else 0
    features["RR_std"] = np.std(rr) if len(rr)>0 else 0
    features["RR_min"] = np.min(rr) if len(rr)>0 else 0
    features["RR_max"] = np.max(rr) if len(rr)>0 else 0
    return features

# --- EMG ---
def extract_emg_features(emg):
    emg_abs = np.abs(emg)
    zero_cross = np.sum(np.diff(np.sign(emg)) != 0)
    rms = np.sqrt(np.mean(emg**2))
    return {
        "EMG_mean": np.mean(emg_abs),
        "EMG_std": np.std(emg),
        "EMG_energy": np.sum(emg**2),
        "EMG_skew": skew(emg),
        "EMG_kurtosis": kurtosis(emg),
        "EMG_rms": rms,
        "EMG_zero_crossings": zero_cross
    }

# --- Respiration ---
def extract_resp_features(resp, fs):
    # Wider bandpass for respiration (0.1-0.8 Hz = 6-48 breaths/min)
    resp_filt = bandpass_filter(resp, fs, 0.1, 0.8)
    
    # Normalize signal
    resp_norm = (resp_filt - np.mean(resp_filt)) / (np.std(resp_filt) + 1e-8)
    
    # Find peaks with adaptive height threshold
    height_thresh = np.percentile(resp_norm, 75)  # Use upper quartile as threshold
    peaks, properties = signal.find_peaks(
        resp_norm,
        distance=int(0.5 * fs),  # Min 0.5s between peaks (allows up to 120 BPM)
        height=height_thresh,
        prominence=0.1  # Ensure peaks are prominent
    )
    
    # Calculate features
    ibi = np.diff(peaks) / fs  # Inter-breath intervals
    
    if len(ibi) > 1:
        br = 60.0 / np.mean(ibi)  # Breaths per minute
        br_std = np.std(ibi)
    else:
        # If insufficient peaks found, try spectral estimation
        freqs, psd = signal.welch(resp_norm, fs=fs, nperseg=min(len(resp_norm), fs*10))
        resp_band = (freqs >= 0.1) & (freqs <= 0.8)  # Respiratory band
        if np.any(resp_band):
            peak_freq = freqs[resp_band][np.argmax(psd[resp_band])]
            br = peak_freq * 60  # Convert Hz to BPM
            br_std = np.std(resp_norm) * 60  # Approximate variation
        else:
            br = 0
            br_std = 0
    
    return {
        "Resp_rate": br,
        "Resp_std": br_std,
        "Resp_power": np.sum(resp_filt**2),
        "Resp_peak_count": len(peaks),
        "Resp_prominence": np.mean(properties["prominences"]) if len(peaks) > 0 else 0
    }

# --- GSR (separate hand/foot) ---
def extract_gsr_features(gsr, fs, prefix="GSR"):
    gsr_filt = lowpass_filter(gsr, fs, 1.0)
    peaks, _ = signal.find_peaks(gsr_filt, distance=fs*1.0, height=np.mean(gsr_filt))
    return {
        f"{prefix}_mean": np.mean(gsr_filt),
        f"{prefix}_std": np.std(gsr_filt),
        f"{prefix}_peaks": len(peaks),
        f"{prefix}_slope": (gsr_filt[-1] - gsr_filt[0]) / len(gsr_filt)
    }

# ------------------------------
# MAIN PIPELINE
# ------------------------------

def process_record(record_path, label, rec_name):
    record = wfdb.rdrecord(record_path)
    signals, sig_names, fs = record.p_signal, record.sig_name, record.fs

    # Resample all signals
    resampled = []
    for sig in signals.T:
        sig_res = resample_signal(sig, fs, RESAMPLE_FS)
        resampled.append(sig_res)
    signals = np.vstack(resampled).T
    fs = RESAMPLE_FS

    seg_len = fs * WINDOW_SIZE
    features_list = []
    annotations_list = []

    for start in range(0, len(signals)-seg_len, seg_len):
        window = signals[start:start+seg_len]
        feats = {}
        for i, name in enumerate(sig_names):
            sig = window[:, i]
            lname = name.lower()
            if "ecg" in lname or "hr" in lname:
                feats.update(extract_ecg_features(sig, fs))
            elif "emg" in lname:
                feats.update(extract_emg_features(sig))
            elif "resp" in lname:
                feats.update(extract_resp_features(sig, fs))
            elif "gsr" in lname:
                if "foot" in lname:
                    feats.update(extract_gsr_features(sig, fs, prefix="GSR_foot"))
                else:
                    feats.update(extract_gsr_features(sig, fs, prefix="GSR_hand"))
        feats["label"] = label
        features_list.append(feats)
        annotations_list.append({"start": start/fs, "end": (start+seg_len)/fs, "label": label})

    df_features = pd.DataFrame(features_list)
    df_annotations = pd.DataFrame(annotations_list)

    # Save separate CSVs
    feat_file = os.path.join(FEATURES_DIR, f"{rec_name}_features.csv")
    ann_file = os.path.join(ANNOTATIONS_DIR, f"{rec_name}_annotations.csv")
    df_features.to_csv(feat_file, index=False)
    df_annotations.to_csv(ann_file, index=False)

    return df_features

# ------------------------------
# RUN PIPELINE
# ------------------------------

records_file = os.path.join(DATA_DIR, "RECORDS")
with open(records_file, 'r') as f:
    record_names = [line.strip() for line in f.readlines() if line.strip()]

all_features = []
for rec in tqdm(record_names):
    rec_path = os.path.join(DATA_DIR, rec)
    rec_name = os.path.basename(rec)
    # Assign label based on your scheme
    import re
    numeric_part = re.findall(r'\d+', rec.replace("drive", ""))[0]
    rec_num = int(numeric_part)
    if rec_num <= 5:
        label = "low"
    elif rec_num <= 10:
        label = "medium"
    else:
        label = "high"
    df = process_record(rec_path, label, rec_name)
    all_features.append(df)

df_all = pd.concat(all_features, ignore_index=True)
df_all.to_csv("stress_features_all.csv", index=False)
print("✅ Features saved per record in 'features/' + annotations in 'annotations/' + combined file 'stress_features_all.csv'")
