import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import find_peaks
import neurokit2 as nk

def extract_features(ecg_signal, fs):
    # 1) Run NeuroKit2 processing
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
    features = nk.ecg_analyze(signals, sampling_rate=fs)

    # 2) Start with main HRV and HR metrics
    summary = {
        "ECG_Rate_Mean":  float(features.get("ECG_Rate_Mean", 0)),
        "HRV_SDNN":       float(features.get("HRV_SDNN", 0)),
        "HRV_RMSSD":      float(features.get("HRV_RMSSD", 0)),
        "HRV_pNN50":      float(features.get("HRV_pNN50", 0)),
    }

    # 3) Manual calculation of QRS duration and QT interval
    q = signals.index[signals["ECG_Q_Peaks"] == 1].to_numpy()
    s = signals.index[signals["ECG_S_Peaks"] == 1].to_numpy()
    t = signals.index[signals["ECG_T_Offsets"] == 1].to_numpy()

    min_len = min(len(q), len(s))
    qrs = ((s[:min_len] - q[:min_len]) / fs * 1000)
    min_len_qt = min(len(q), len(t))
    qt = ((t[:min_len_qt] - q[:min_len_qt]) / fs * 1000)

    summary["ECG_QRS_Duration"] = float(np.mean(qrs)) if qrs.size > 0 else np.nan
    summary["ECG_QT_Interval"]  = float(np.mean(qt))  if qt.size > 0  else np.nan

    return summary

st.set_page_config(layout="wide")
st.title("🔬 ECG Signal Diagnostic Dashboard")

# Sidebar selections
st.sidebar.header("🔧 Select Signal Parameters")
subject = st.sidebar.selectbox("Subject", [f"sub{str(i).zfill(2)}" for i in range(1, 11)])
level = st.sidebar.selectbox("Level", [f"l{i}" for i in range(1, 6)])
signal_type = st.sidebar.selectbox("Signal Type", ["fecg", "mecg"])

# Construct file path
filename = f"{subject}_{level}_c0_combined_cleaned_{signal_type}.npy"
filepath = f"ica_cleaned_signals/{filename}"

# Load signal
fs = 250  # Update if needed
signal = np.load(filepath)
duration = len(signal) / fs

# Plotting duration control
seconds = st.sidebar.slider("Duration to display (s)", 1, int(duration), 10)
samples = int(seconds * fs)

# Layout
col1, col2 = st.columns([2, 1])

# ==== Left: Signal Plot ====
with col1:
    st.subheader("📉 ECG Signal with R-peaks")
    fig, ax = plt.subplots()
    t = np.arange(samples) / fs
    ax.plot(t, signal[:samples], label="ECG")
    peaks, _ = find_peaks(signal[:samples], distance=fs*0.4)
    ax.plot(peaks / fs, signal[peaks], "ro", label="R-peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend()
    st.pyplot(fig)

# ==== Right: Feature Table ====
with col2:
    st.subheader("Extracted Features")
    features = extract_features(signal, fs)

    # Descriptions and ranges (fetal vs maternal)
    info = {
        "ECG_Rate_Mean": {
            "desc": "Average heart rate over the signal",
            "normal_fetal": "110–160 bpm",
            "normal_maternal": "60–100 bpm"
        },
        "HRV_SDNN": {
            "desc": "Standard deviation of RR intervals",
            "normal_fetal": "< 20 ms",
            "normal_maternal": "30–50 ms"
        },
        "HRV_RMSSD": {
            "desc": "Root mean square of successive RR differences",
            "normal_fetal": "5–25 ms",
            "normal_maternal": "20–50 ms"
        },
        "HRV_pNN50": {
            "desc": "% of RR intervals differing > 50 ms",
            "normal_fetal": "Low (often <10%)",
            "normal_maternal": "High (often >30%)"
        },
        "ECG_QRS_Duration": {
            "desc": "Time of ventricular depolarization",
            "normal_fetal": "50–80 ms",
            "normal_maternal": "80–100 ms"
        },
        "ECG_QT_Interval": {
            "desc": "Total time for ventricular activity",
            "normal_fetal": "140–220 ms",
            "normal_maternal": "350–440 ms"
        }
    }

    # Show metrics with abnormal flagging
flags = []
for key, value in features.items():
    norm = info.get(key, {})
    if signal_type == "fecg":
        normal_range = norm.get("normal_fetal", "-")
        if key == "ECG_Rate_Mean":
            abnormal = value < 110 or value > 160
        elif key == "ECG_QRS_Duration":
            abnormal = value < 50 or value > 80
        elif key == "ECG_QT_Interval":
            abnormal = value < 140 or value > 220
        elif key == "HRV_SDNN":
            abnormal = value > 20
        elif key == "HRV_RMSSD":
            abnormal = value < 5 or value > 25
        elif key == "HRV_pNN50":
            abnormal = value > 10
        else:
            abnormal = False
    else:  # mecg
        normal_range = norm.get("normal_maternal", "-")
        if key == "ECG_Rate_Mean":
            abnormal = value < 60 or value > 100
        elif key == "ECG_QRS_Duration":
            abnormal = value > 120
        elif key == "ECG_QT_Interval":
            abnormal = value > 450
        elif key == "HRV_SDNN":
            abnormal = value < 30 or value > 50
        elif key == "HRV_RMSSD":
            abnormal = value < 20 or value > 50
        elif key == "HRV_pNN50":
            abnormal = value < 30
        else:
            abnormal = False

    flags.append(abnormal)

    color = "🟢" if not abnormal else "🔴"
    st.markdown(f"""
    **{color} {key}:** {value:.2f}  
    _{norm.get("desc", "No description available.")}_  
    **Expected ({signal_type.upper()}):** {normal_range}
    """)
    # Summary alert
    if any(flags):
        st.error("⚠️ Some metrics are outside the normal range. This signal may need clinical review.")
    else:
        st.success("✅ All metrics are within expected physiological ranges.")


# Footer
st.markdown("---")

