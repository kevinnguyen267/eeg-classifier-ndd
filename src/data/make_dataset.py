from glob import glob

import asrpy
import mne
import numpy as np
import pandas as pd
from mne_icalabel import label_components


def get_subjects(file):
    participants = pd.read_csv(file, sep="\t")
    return participants[["participant_id", "Group"]].set_index("participant_id")


def read_raw_eeg(file):
    return mne.io.read_raw_eeglab(file)


def preprocess(raw):
    # Apply Butterworth bandpass filter (0.5-45 Hz range) to filter out non-physiological noise
    raw_filtered = raw.copy()
    iir_params = dict(order=4, ftype="butter")
    raw_filtered.filter(l_freq=0.5, h_freq=45, method="iir", iir_params=iir_params)

    # Artifact Subspace Reconstruction (ASR) routine to filter out persistent or extreme outlier artifacts
    asr = asrpy.ASR(sfreq=raw_filtered.info["sfreq"], cutoff=17, win_len=0.5)
    asr.fit(raw_filtered)
    raw_asr = asr.transform(raw_filtered)

    # Independent Component Analysis (ICA) method to filter out eye/jaw artifacts
    ica = mne.preprocessing.ICA(
        n_components=len(raw_asr.info["ch_names"]), method="infomax"
    )
    ica.fit(raw_asr)
    labels = label_components(raw_asr, ica, "iclabel")
    labels_to_remove = ["eye blink", "muscle artifact"]
    ica.exclude = [
        i for i, label in enumerate(labels["labels"]) if label in labels_to_remove
    ]
    return ica.apply(raw_asr)


def epoch(data):
    # Epoch Segmentation
    events = mne.make_fixed_length_events(data, duration=4, overlap=0.5)
    return mne.Epochs(data, events, tmin=0, tmax=4, baseline=None)


def extract_alpha_rbp(epochs, sf):
    x = epochs.get_data()
    fmin = 0.5
    fmax = 45
    win = int(2 / fmin * sf)
    # Obtain Power Spectral Density (PSD) using the Welch method over the whole frequency range of interest (0.5-45 Hz)
    psds, freqs = mne.time_frequency.psd_array_welch(
        x, sf, fmin=fmin, fmax=fmax, n_fft=win
    )

    # Calculate Relative Band Power (RBP) of the Alpha band (8-13 Hz)
    alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
    alpha_power_total = np.mean(np.sum(psds[:, :, alpha_idx], axis=2))
    all_bands_power_total = np.mean(np.sum(psds[:, :, :], axis=2))
    return alpha_power_total / all_bands_power_total


df = get_subjects("../../data/raw/openneuro-ds004504/participants.tsv")

files = glob("../../data/raw/openneuro-ds004504/*/*/*.set")
for f in files:
    subject_id = f.split("/")[5]

    raw = read_raw_eeg(f)
    data = preprocess(raw)
    epochs = epoch(data)
    alpha_rbp = extract_alpha_rbp(epochs, raw.info["sfreq"])

    df.loc[subject_id, "alpha_rbp"] = alpha_rbp

df.to_pickle("../../data/interim/preprocessed.pkl")
