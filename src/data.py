import os
import glob
import pickle

import numpy as np
import pandas as pd

# (pickle name, code used in the result tables), in the usual participant order
PARTICIPANTS = [
    ("Amin", "AMB"), ("Arya", "ARM"), ("Eisa", "EIA"), ("Elahe", "ELD"),
    ("Farzam", "FAS"), ("Javad", "JAY"), ("Mehrnaz", "MEB"), ("Nastaran", "NAD"),
    ("Razieh", "RAS"), ("Sajad", "SAS"), ("Shahrzad", "SHU"), ("Yaser", "AYB"),
]

N_SENTENCES = 165
N_CHANNELS = 126


def load_labels(labels_dir):
    files = sorted(glob.glob(os.path.join(labels_dir, "Total_Sentiment_Raw_*.xlsx")))
    if not files:
        raise FileNotFoundError(f"no Total_Sentiment_Raw_*.xlsx under {labels_dir}")
    df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
    df = df[["id", "sentence_per", "sentiment_label"]].copy()
    df["id"] = df["id"].astype(int)
    df["sentiment_label"] = df["sentiment_label"].astype(int)
    return df.sort_values("id").reset_index(drop=True)


def _read_subject(trt_dir, name):
    with open(os.path.join(trt_dir, f"{name}_trt_total.pickle"), "rb") as f:
        return pickle.load(f)


def load_eeg(trt_dir):
    """Word-level total-reading-time EEG as (subjects, sentences, words, channels)."""
    subjects = [_read_subject(trt_dir, name) for name, _ in PARTICIPANTS]
    max_words = max(max(len(s[t]["word"]) for t in s) for s in subjects)

    eeg = np.zeros((len(subjects), N_SENTENCES, max_words, N_CHANNELS), dtype=np.float32)
    for k, sub in enumerate(subjects):
        for t in sub:
            sid = sub[t]["sentenceId"]               # 1-based, lines up with the label 'id'
            for wi, word in sub[t]["word"].items():
                vec = np.asarray(word["TRT_total"])
                # un-fixated words are stored as a length-2 stub; leave those at zero
                if vec.shape == (N_CHANNELS,):
                    eeg[k, sid - 1, wi] = vec
    return eeg


def build_binary_dataset(trt_dir, labels_dir):
    """Positive vs negative sentences (neutral dropped -> 110 sentences)."""
    labels = load_labels(labels_dir)
    eeg = load_eeg(trt_dir)

    binary = labels[labels.sentiment_label != 0].sort_values("id").copy()
    binary["label_id"] = binary["sentiment_label"].map({-1: 0, 1: 1})

    X = eeg[:, binary["id"].to_numpy() - 1]
    y = binary["label_id"].to_numpy()
    codes = [c for _, c in PARTICIPANTS]
    return X, y, codes
