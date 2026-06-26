import os
import random
import argparse

import numpy as np
import pandas as pd

from data import build_binary_dataset
from model import build_cnn


def _seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def run(trt_dir, labels_dir, out_dir="results", epochs=100, batch_size=64, seed=45):
    _seed_everything(seed)

    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import classification_report

    X, y, codes = build_binary_dataset(trt_dir, labels_dir)
    X = X[..., None]                       # conv2d wants a channel axis
    Y = to_categorical(y, num_classes=2)
    n = X.shape[0]

    rows = []
    for i in range(n):
        # leave one participant out
        x_test = X[i]
        x_train = np.concatenate([X[j] for j in range(n) if j != i], axis=0)
        y_train = np.concatenate([Y for j in range(n) if j != i], axis=0)

        tf.keras.backend.clear_session()
        model = build_cnn(X.shape[2:])
        stop = EarlyStopping(monitor="val_accuracy", patience=40, mode="max",
                             restore_best_weights=True, start_from_epoch=10)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_split=0.15, callbacks=[stop], verbose=0)

        pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        macro = classification_report(y, pred, output_dict=True, zero_division=0)["macro avg"]
        rows.append({"participant": codes[i],
                     "precision": macro["precision"],
                     "recall": macro["recall"],
                     "f1": macro["f1-score"]})
        print(f"{codes[i]:>4}  f1={macro['f1-score']:.3f}")

    results = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "per_participant_eeg.csv")
    results.to_csv(out_path, index=False)

    print(f"\nmean macro-f1: {results['f1'].mean():.4f}")
    print(f"saved -> {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser(
        description="EEG-only binary sentiment, leave-one-participant-out (TeCo task 1).")
    ap.add_argument("--trt-dir", required=True, help="folder with the *_trt_total.pickle files")
    ap.add_argument("--labels-dir", required=True, help="folder with Total_Sentiment_Raw_*.xlsx")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=45)
    args = ap.parse_args()

    run(args.trt_dir, args.labels_dir, args.out_dir,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":
    main()
