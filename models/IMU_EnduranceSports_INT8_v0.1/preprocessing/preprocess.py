from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------
# 1. FILE PATHS
# ---------------------------------------
# This file lives in models/<version>/preprocessing.
VERSION_DIR = Path(__file__).resolve().parent.parent

RAW_CSV = (
    VERSION_DIR
    / "preprocessing"
    / "raw"
    / "labeled_imu"
    / "generalized_sports_imu_labeled.csv"
)

# Temporary compatibility path while the existing raw folder is still open.
if not RAW_CSV.exists():
    RAW_CSV = (
        VERSION_DIR.parents[1]
        / "labeled_imu"
        / "generalized_sports_imu_labeled.csv"
    )

OUTPUT_DIR = VERSION_DIR / "preprocessing" / "processed" / "cnn_preprocessed"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cnn_imu_windows.npz"


# ---------------------------------------
# 2. SETTINGS
# ---------------------------------------
SENSOR_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]

CNN_INPUT_COLUMNS = SENSOR_COLUMNS + [
    "acc_magnitude",
    "gyro_magnitude"
]

GROUP_COLUMNS = [
    "activity",
    "subject",
    "window_id"
]

WINDOW_SIZE = 125

# 100000 is divisible by 125.
# Therefore a complete window stays in one chunk.
CHUNK_SIZE = 100_000


def create_cnn_windows():
    """
    Converts raw sensor rows into CNN input windows.

    One window:
    125 rows × 6 IMU values
    = 5 seconds × 25 Hz
    """

    sequences = []
    labels = []
    subjects = []
    window_ids = []

    use_columns = GROUP_COLUMNS + SENSOR_COLUMNS

    print("Step 1: Reading raw IMU dataset...")

    for chunk in pd.read_csv(
        RAW_CSV,
        usecols=use_columns,
        chunksize=CHUNK_SIZE
    ):

        chunk["acc_magnitude"] = np.sqrt(
            chunk["ax"] ** 2
            + chunk["ay"] ** 2
            + chunk["az"] ** 2
        )

        chunk["gyro_magnitude"] = np.sqrt(
            chunk["gx"] ** 2
            + chunk["gy"] ** 2
            + chunk["gz"] ** 2
        )
        grouped = chunk.groupby(
            GROUP_COLUMNS,
            sort=False
        )

        for (activity, subject, window_id), group in grouped:

            # Keep only complete 5-second windows.
            if len(group) != WINDOW_SIZE:
                continue

            imu_window = group[
                CNN_INPUT_COLUMNS
            ].to_numpy(dtype=np.float32)

            sequences.append(imu_window)
            labels.append(activity)
            subjects.append(subject)
            window_ids.append(window_id)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels)
    subject_array = np.array(subjects)
    window_id_array = np.array(window_ids)

    return X, y, subject_array, window_id_array


def main():

    X, y, subjects, window_ids = create_cnn_windows()

    print("\nPreprocessing completed.")

    print("\nCNN input shape:")
    print(X.shape)

    print("\nExpected format:")
    print("(number of samples, 125 time steps, 8 IMU values)")

    print("\nActivity classes:")
    unique_labels, counts = np.unique(
        y,
        return_counts=True
    )

    for activity, count in zip(unique_labels, counts):
        print(f"{activity}: {count}")

    # Save CNN data.
    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        subjects=subjects,
        window_ids=window_ids
    )

    print("\nSaved CNN preprocessed data:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
