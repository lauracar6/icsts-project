# %%
import neurokit2 as nk
import numpy as np
from scipy.signal import butter, filtfilt
import json
import os

# %%
def pair_peaks(peaks1, peaks2, max_distance=200):
    pairs1 = []
    pairs2 = []
    j = 0
    for i in range(len(peaks1)):
        while j < len(peaks2) and peaks2[j] < peaks1[i]:
            j += 1
        if j < len(peaks2) and abs(peaks2[j] - peaks1[i]) <= max_distance:
            pairs1.append(peaks1[i])
            pairs2.append(peaks2[j])
            j += 1
    return np.array(pairs1), np.array(pairs2)

def is_sinus_rhythm(p_peaks, r_peaks, fs, rr_threshold=0.2):
    if len(p_peaks) < 3 or len(r_peaks) < 3:
        return False

    rr_intervals = np.diff(r_peaks) / fs
    if np.std(rr_intervals) > rr_threshold:
        return False

    matched = 0
    for r in r_peaks:
        preceding_p = p_peaks[p_peaks < r]
        if len(preceding_p) > 0 and r - preceding_p[-1] < fs * 0.2:
            matched += 1

    return matched / len(r_peaks) > 0.8

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# %%
def check_signal_quality(ecg_signal, fs, min_r_peaks=3, min_std=0.05):
    try:
        ecg_signal = bandpass_filter(ecg_signal, fs)
        signals, _ = nk.ecg_process(ecg_signal, sampling_rate=fs)
        r_peaks = signals.index[signals["ECG_R_Peaks"] == 1].to_numpy()
        num_r_peaks = len(r_peaks)
        signal_std = np.std(ecg_signal)
        return num_r_peaks >= min_r_peaks and signal_std >= min_std
    except Exception as e:
        print(f"Signal quality check failed: {e}")
        return False


# %%
def extract_features(ecg_signal, fs, subject_type="fetal"):
    summary = {
        "Heart_Rate_Mean": np.nan,
        "PR_Interval_ms": np.nan,
        "QRS_Duration_ms": np.nan,
        "QT_Interval_ms": np.nan,
        "Sinus_Rhythm": False,
        "Quality_OK": False,
    }

    try:
        # Step 1: Filter the signal
        filtered_signal = bandpass_filter(ecg_signal, fs)

        # Step 2: Check signal quality
        quality_ok = check_signal_quality(filtered_signal, fs)
        summary["Quality_OK"] = quality_ok

        if not quality_ok:
            return summary  # Skip processing if poor quality

        # Step 3: Process with neurokit
        signals, info = nk.ecg_process(filtered_signal, sampling_rate=fs)
        features = nk.ecg_analyze(signals, sampling_rate=fs)
        summary["Heart_Rate_Mean"] = float(features.get("ECG_Rate_Mean", np.nan))

        # Step 4: Extract peaks
        q = signals.index[signals["ECG_Q_Peaks"] == 1].to_numpy()
        s = signals.index[signals["ECG_S_Peaks"] == 1].to_numpy()
        p = signals.index[signals["ECG_P_Peaks"] == 1].to_numpy()
        t = signals.index[signals["ECG_T_Offsets"] == 1].to_numpy()
        r = signals.index[signals["ECG_R_Peaks"] == 1].to_numpy()

        # Step 5: Intervals
        if len(q) > 0 and len(s) > 0:
            q_matched, s_matched = pair_peaks(q, s)
            if len(q_matched) > 0:
                qrs = ((s_matched - q_matched) / fs * 1000)
                summary["QRS_Duration_ms"] = float(np.mean(qrs))

        if len(p) > 0 and len(q) > 0:
            p_matched, q_matched = pair_peaks(p, q)
            if len(p_matched) > 0:
                pr = ((q_matched - p_matched) / fs * 1000)
                summary["PR_Interval_ms"] = float(np.mean(pr))

        if len(q) > 0 and len(t) > 0:
            q_matched, t_matched = pair_peaks(q, t)
            if len(q_matched) > 0:
                qt = ((t_matched - q_matched) / fs * 1000)
                summary["QT_Interval_ms"] = float(np.mean(qt))

        # Step 6: Sinus rhythm check
        summary["Sinus_Rhythm"] = is_sinus_rhythm(p, r, fs)

    except Exception as e:
        print(f"⚠️ Feature extraction failed: {e}")

    return summary


# %%
def sanitize_for_json(data):
    """Convert NumPy types to native Python types recursively for JSON serialization."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, np.generic):
        return data.item()
    else:
        return data


base_dir = "ica_cleaned_signals"
output_dir = "metrics_json"
os.makedirs(output_dir, exist_ok=True)

files = [f for f in os.listdir(base_dir) if f.endswith(".npy")]
fs = 250  # sampling frequency

for filename in files:
    filepath = os.path.join(base_dir, filename)

    if "fecg" in filename:
        subject_type = "fetal"
    elif "mecg" in filename:
        subject_type = "maternal"
    else:
        print(f"Skipping unknown type for file {filename}")
        continue

    signal = np.load(filepath)

    try:
        features = extract_features(signal, fs, subject_type)

        # Add metadata
        features["subject_type"] = subject_type
        features["filename"] = filename

        # Convert all values to JSON-safe types
        json_safe_features = sanitize_for_json(features)

        # Save to JSON
        json_filename = filename.replace(".npy", "_features.json")
        json_path = os.path.join(output_dir, json_filename)

        with open(json_path, "w") as json_file:
            json.dump(json_safe_features, json_file, indent=4)

        print(f"Saved features for {filename} to {json_filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")


# %%



