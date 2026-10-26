from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "trained_cnn_model"
PREPROCESSED_FILE = PROJECT_DIR / "cnn_preprocessed" / "cnn_imu_windows.npz"

KERAS_MODEL = MODEL_DIR / "sports_activity_cnn.keras"
H5_MODEL = MODEL_DIR / "sports_activity_cnn.h5"
INT8_TFLITE_MODEL = MODEL_DIR / "sports_activity_cnn_int8.tflite"
NORMALIZATION_FILE = MODEL_DIR / "normalization_values.npz"


def main():
    print("Loading the saved Keras model...")
    model = tf.keras.models.load_model(KERAS_MODEL)

    print("Exporting EdgeSphere-compatible HDF5 model...")
    model.save(H5_MODEL, include_optimizer=False)

    print("Preparing representative IMU data for INT8 calibration...")
    raw_data = np.load(PREPROCESSED_FILE, allow_pickle=False)
    raw_windows = raw_data["X"].astype(np.float32)

    normalization = np.load(NORMALIZATION_FILE)
    mean = normalization["mean"].astype(np.float32)
    std = normalization["std"].astype(np.float32)

    normalized_windows = (raw_windows - mean) / std
    calibration_windows = normalized_windows[:500]

    def representative_dataset():
        for window in calibration_windows:
            yield [np.expand_dims(window, axis=0).astype(np.float32)]

    print("Converting to fully INT8 TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    int8_tflite = converter.convert()
    INT8_TFLITE_MODEL.write_bytes(int8_tflite)

    print("Checking the generated INT8 TensorFlow Lite model...")
    interpreter = tf.lite.Interpreter(model_path=str(INT8_TFLITE_MODEL))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    if input_info["dtype"] != np.int8 or output_info["dtype"] != np.int8:
        raise RuntimeError("INT8 conversion did not produce INT8 input and output tensors.")

    print("\nExport completed successfully.")
    print("H5 model:", H5_MODEL)
    print("INT8 TFLite model:", INT8_TFLITE_MODEL)
    print("TFLite input shape:", input_info["shape"])
    print("TFLite input dtype:", input_info["dtype"])
    print("TFLite output dtype:", output_info["dtype"])


if __name__ == "__main__":
    main()
