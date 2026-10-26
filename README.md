# IMU-Based Sports and Endurance Activity Detection

A 1D-CNN model that classifies five-second IMU windows into eight sports and
endurance activities using accelerometer and gyroscope signals.

## Activities

- Basketball
- CrossTrainer
- CyclingHorizontal
- Jumping
- Rowing
- Running
- Stepper
- WalkingFlat

## Dataset

The project uses a cleaned, labelled subset of the UCI Daily and Sports
Activities (DSA) dataset. The raw dataset is not included in this repository.

Source: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

The selected data is sampled at 25 Hz. Each model input represents a
five-second window containing 125 time steps.

## Input features

The CNN receives an input of shape `(125, 8)` for each window:

`ax, ay, az, gx, gy, gz, acc_magnitude, gyro_magnitude`

The first six are the 3-axis accelerometer and 3-axis gyroscope signals. The
last two are derived accelerometer and gyroscope magnitudes.

## Current experiment

Subject split:

- Training: p1-p4, p7-p8
- Validation: p5
- Final testing: p6

Results:

- Validation accuracy: 97.71%
- Final test accuracy: 95.25%

## Repository structure

```text
models/
└── IMU_EnduranceSports_INT8_v0.1/
    ├── model_card.yaml
    ├── model/             # INT8 TFLite deployment model and supporting artifacts
    ├── preprocessing/     # Preprocessing code, local raw data and derived windows
    ├── training/          # Training and format-export scripts
    ├── evaluation/        # Metrics and confusion-matrix figures
    └── release/           # Generated release documents
```

## Model artifacts

- `IMU_EnduranceSports_INT8_v0.1.h5` — Keras H5 model
- `IMU_EnduranceSports_INT8_v0.1.tflite` — fully INT8 TensorFlow Lite deployment model
- `normalization_values.npz` — training normalization values
- `label_encoder.pkl` — output-class label encoder

## Notes

The model was evaluated using a subject-wise split, so validation and test
subjects were not included in training.
