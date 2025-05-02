import streamlit as st
import os
import numpy as np
import json
import matplotlib.pyplot as plt

# --- Constants ---
SIGNAL_DIR = "ica_cleaned_signals"
FEATURES_DIR = "metrics_json"
fs = 250
duration = 5

# --- Metric Info ---
METRIC_INFO = {
    "Heart_Rate_Mean": {
        "range_fetal": "110-160 bpm",
        "range_maternal": "60-100 bpm",
        "desc": "Average heart rate over the signal."
    },
    "PR_Interval_ms": {
        "range_fetal": "110-130 ms",
        "range_maternal": "120-200 ms",
        "desc": "Time from onset of P wave to start of QRS complex."
    },
    "QRS_Duration_ms": {
        "range_fetal": "40-60 ms",
        "range_maternal": "60-120 ms",
        "desc": "Duration of ventricular depolarization."
    },
    
}

# --- Extract all filenames ---
files = [f for f in os.listdir(SIGNAL_DIR) if f.endswith(".npy")]

# --- Parse metadata from filenames ---
parsed = []
for f in files:
    parts = f.split("_")
    if len(parts) >= 6:
        subject = parts[0]       # sub01
        level = parts[1]         # l1
        signal_type = parts[-1].replace(".npy", "")  # fecg or mecg
        parsed.append((subject, level, signal_type, f))

subjects = sorted(set(p[0] for p in parsed))
levels = sorted(set(p[1] for p in parsed))
types = sorted(set(p[2] for p in parsed))

# --- Streamlit UI ---
st.title("🫀 ECG Signal Dashboard")

selected_subject = st.selectbox("Select Subject", subjects)
selected_level = st.selectbox("Select Level", levels)
selected_type = st.selectbox("Select Signal Type", types)

# --- Match file ---
matching_files = [p for p in parsed if p[0] == selected_subject and p[1] == selected_level and p[2] == selected_type]

if matching_files:
    filename = matching_files[0][3]
    signal_path = os.path.join(SIGNAL_DIR, filename)

    signal = np.load(signal_path)
    cropped = signal[:fs * duration]
    time = np.arange(len(cropped)) / fs

    # Plot ECG
    st.subheader("📈 ECG Plot (first 5 seconds)")
    duration_sec = 3  # Show 3 seconds for better peak visibility
    samples_to_plot = fs * duration_sec
    time_axis = np.arange(samples_to_plot) / fs

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, signal[:samples_to_plot], color="blue", linewidth=1.5)

    ax.set_title("ECG Signal (First 3 Seconds)", fontsize=14)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")

    # Add grid like ECG paper
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor("#f9f9f9")

    st.pyplot(fig)

    # Load and show metrics
    metrics_filename = filename.replace(".npy", "_features.json")
    metrics_path = os.path.join(FEATURES_DIR, metrics_filename)

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            features = json.load(f)

        subject_type = features.get("subject_type", "fetal" if "fecg" in filename else "maternal")

        st.subheader("ECG Metrics")
        for key, value in features.items():
            if key in METRIC_INFO:
                st.markdown(f"**{key.replace('_', ' ')}**: `{value}`")

                # Show healthy range
                range_key = "range_fetal" if subject_type == "fetal" else "range_maternal"
                st.markdown(f"- 🟢 *Healthy Range* ({subject_type}): `{METRIC_INFO[key][range_key]}`")

                # Show description
                st.markdown(f"- {METRIC_INFO[key]['desc']}")

    else:
        st.warning("Metrics file not found.")
else:
    st.warning("No matching file found for the selected combination.")