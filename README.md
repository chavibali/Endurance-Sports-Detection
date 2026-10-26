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
preprocess_cnn.py          # Converts labelled raw IMU data into CNN windows
train_cnn_model.py         # Trains and evaluates the 1D-CNN
export_model_formats.py    # Exports the trained model to H5 and INT8 TFLite
trained_cnn_model/         # Model, labels, normalization data and evaluation plots
```

## Model artifacts

- `sports_activity_cnn.h5` — Keras H5 model
- `sports_activity_cnn_int8.tflite` — fully INT8 TensorFlow Lite deployment model
- `normalization_values.npz` — training normalization values
- `label_encoder.pkl` — output-class label encoder

## Notes

The model was evaluated using a subject-wise split, so validation and test
subjects were not included in training.
