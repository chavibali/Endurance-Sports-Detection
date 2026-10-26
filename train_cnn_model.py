from pathlib import Path

import joblib
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------
# 1. FILE PATHS
# ---------------------------------------
# This file is directly inside project root.
PROJECT_DIR = Path(__file__).resolve().parent

PREPROCESSED_FILE = (
    PROJECT_DIR
    / "cnn_preprocessed"
    / "cnn_imu_windows.npz"
)

MODEL_DIR = PROJECT_DIR / "trained_cnn_model"
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------
# 2. SETTINGS
# ---------------------------------------
WINDOW_SIZE = 125
NUMBER_OF_IMU_FEATURES = 8

tf.keras.utils.set_random_seed(42)


# ---------------------------------------
# 3. CNN ARCHITECTURE
# No GlobalAveragePooling / GlobalMaxPooling
# ---------------------------------------
def build_cnn(number_of_classes):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(WINDOW_SIZE, NUMBER_OF_IMU_FEATURES)
        ),

        # Block 1
        tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=5,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        # Block 2
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        # Block 3
        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        # No global pooling is used.
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            90,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            number_of_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ---------------------------------------
# 4. LOAD AND SPLIT DATA
# ---------------------------------------
def main():

    print("Step 1: Loading CNN preprocessed data...")

    loaded_data = np.load(
        PREPROCESSED_FILE,
        allow_pickle=False
    )

    X = loaded_data["X"]
    y_text = loaded_data["y"]
    subjects = loaded_data["subjects"]

    print("CNN input shape:", X.shape)

        # ---------------------------------------
    # SUBJECT-WISE EXPERIMENTAL TRAIN / VALIDATION / TEST SPLIT
    # p1-p4, p7, p8 = training | p5 = validation | p6 = testing
    # ---------------------------------------
    train_mask = np.isin(
        subjects,
        ["p1", "p2", "p3", "p4", "p7", "p8"]
    )

    validation_mask = subjects == "p5"
    test_mask = subjects == "p6"

    X_train = X[train_mask]
    X_validation = X[validation_mask]
    X_test = X[test_mask]

    y_train_text = y_text[train_mask]
    y_validation_text = y_text[validation_mask]
    y_test_text = y_text[test_mask]

    print("\nSubject-wise experimental split completed:")
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_validation))
    print("Test samples:", len(X_test))

    # ---------------------------------------
    # 5. NORMALIZATION
    # Fit only on training data.
    # ---------------------------------------
    train_mean = X_train.mean(
        axis=(0, 1),
        keepdims=True
    )

    train_std = X_train.std(
        axis=(0, 1),
        keepdims=True
    )

    train_std[train_std == 0] = 1

    X_train = (X_train - train_mean) / train_std
    X_validation = (
        X_validation - train_mean
    ) / train_std

    X_test = (X_test - train_mean) / train_std

    # ---------------------------------------
    # 6. LABEL ENCODING
    # ---------------------------------------
    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        y_train_text
    )

    y_validation = label_encoder.transform(
        y_validation_text
    )

    y_test = label_encoder.transform(
        y_test_text
    )

    print("\nActivity classes:")
    print(list(label_encoder.classes_))

    # ---------------------------------------
    # 7. TRAIN CNN
    # ---------------------------------------
    model = build_cnn(
        number_of_classes=len(label_encoder.classes_)
    )

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    print("\nStep 2: Training 1D-CNN model...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation
        ),
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )

    # ---------------------------------------
    # 8. VALIDATION RESULT
    # ---------------------------------------
    _, validation_accuracy = model.evaluate(
        X_validation,
        y_validation,
        verbose=0
    )

    print(
        "\nValidation Accuracy:",
        round(validation_accuracy * 100, 2),
        "%"
    )

    # Validation class-wise report: only p7 subject
    validation_probabilities = model.predict(
        X_validation,
        verbose=0
    )

    validation_predictions = np.argmax(
        validation_probabilities,
        axis=1
    )

    validation_report = classification_report(
        y_validation,
        validation_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    print("\nValidation Classification Report:\n")
    print(validation_report)

        # ---------------------------------------
    # VALIDATION CONFUSION MATRIX
    # ---------------------------------------
    figure, axis = plt.subplots(
        figsize=(11, 8)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_validation,
        validation_predictions,
        display_labels=label_encoder.classes_,
        xticks_rotation=45,
        ax=axis
    )

    plt.title("Validation Confusion Matrix - Subject p5")
    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / "validation_confusion_matrix_p7.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------------
    # 9. FINAL TEST RESULT
    # ---------------------------------------
    test_probabilities = model.predict(
        X_test,
        verbose=0
    )

    test_predictions = np.argmax(
        test_probabilities,
        axis=1
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    report = classification_report(
        y_test,
        test_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    print(
        "\nFinal Test Accuracy:",
        round(test_accuracy * 100, 2),
        "%"
    )

    print("\nClassification Report:\n")
    print(report)

    # ---------------------------------------
    # 10. SAVE MODEL AND RESULTS
    # ---------------------------------------
    model.save(
        MODEL_DIR / "sports_activity_cnn.keras"
    )

    joblib.dump(
        label_encoder,
        MODEL_DIR / "label_encoder.pkl"
    )

    np.savez(
        MODEL_DIR / "normalization_values.npz",
        mean=train_mean,
        std=train_std
    )

    with open(
        MODEL_DIR / "cnn_results.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Final Test Accuracy: "
            f"{test_accuracy * 100:.2f}%\n\n"
        )

        file.write(report)

    # ---------------------------------------
    # 11. CONFUSION MATRIX
    # ---------------------------------------
    figure, axis = plt.subplots(
        figsize=(11, 8)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        display_labels=label_encoder.classes_,
        xticks_rotation=45,
        ax=axis
    )

    plt.title("1D-CNN Sports Activity Detection")
    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / "cnn_confusion_matrix.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------------
    # 12. ACCURACY GRAPH
    # ---------------------------------------
    plt.figure(figsize=(9, 6))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / "cnn_accuracy_graph.png",
        dpi=200
    )

    plt.close()

    print("\nCNN training completed.")
    print("Saved model and results in:")
    print(MODEL_DIR)


if __name__ == "__main__":
    main()
