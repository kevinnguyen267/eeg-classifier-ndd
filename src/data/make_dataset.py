from glob import glob

import asrpy
import mne
import numpy as np
import pandas as pd
from mne_icalabel import label_components


def get_subjects(file):
    participants = pd.read_csv(file, sep="\t")
    return (
        participants[["participant_id", "Group"]]
        .set_index("participant_id")
        .rename_axis("subject_id")
    )


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

    # Independent Component Analysis (ICA) method to filter out eye/muscle artifacts
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


def extract_rbps(epochs, sfreq, win, bands):
    x = epochs.get_data()
    # Obtain Power Spectral Density (PSD) using the Welch method over the whole frequency range of interest (0.5-45 Hz)
    psds, freqs = mne.time_frequency.psd_array_welch(
        x, sfreq, fmin=0.5, fmax=45, n_fft=win
    )

    # Calculate total band power (mean over epochs and channels)
    total_power = np.mean(np.sum(psds, axis=2))

    rbps = {}
    # Calculate Relative Band Power (RBP) for each band
    # psds shape (n_epochs, n_channels, n_freqs)
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.mean(np.sum(psds[:, :, idx], axis=2))
        rbps[band] = band_power / total_power

    return rbps


# Intialize DataFrame with subject data
df = get_subjects("../../data/raw/openneuro-ds004504/participants.tsv")

# Preprocess and segment into epochs
files = glob("../../data/raw/openneuro-ds004504/*/*/*.set")
for f in files:
    subject_id = f.split("/")[5]

    raw = read_raw_eeg(f)
    data = preprocess(raw)
    epochs = epoch(data)
    epochs.save(f"../../data/interim/epochs/{subject_id}-epo.fif", overwrite=True)


# Extract features from epochs
sfreq = read_raw_eeg(files[0]).info["sfreq"]
win = int(2 / 0.5 * sfreq)
files = glob("../../data/interim/epochs/*epo.fif")
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 25),
    "gamma": (25, 45),
}
for f in files:
    subject_id = f.split("/")[5].rstrip("-epo.fif")
    epochs = mne.read_epochs(f)
    rbps = extract_rbps(epochs, sfreq, win, bands)
    for band, rbp in rbps.items():
        df.loc[subject_id, f"{band}_rbp"] = rbp

df.to_pickle("../../data/interim/01_data_processed.pkl")
