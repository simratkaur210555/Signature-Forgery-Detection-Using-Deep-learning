import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    auc
)

from .dataset import create_dataset_splits
from .preprocessing import preprocess_signature
from .model import create_siamese_model
from .config import MODEL_PATH, BATCH_SIZE


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD IMAGE PAIRS
# ============================================================

def load_pairs(pairs):

    images_a = []
    images_b = []
    labels = []

    for index, (path_a, path_b, label) in enumerate(pairs):

        try:

            image_a = preprocess_signature(
                path_a
            )

            image_b = preprocess_signature(
                path_b
            )

            images_a.append(image_a)

            images_b.append(image_b)

            labels.append(label)

        except Exception as error:

            print(
                f"Skipping pair {index}: {error}"
            )

    return (
        np.asarray(
            images_a,
            dtype=np.float32
        ),

        np.asarray(
            images_b,
            dtype=np.float32
        ),

        np.asarray(
            labels,
            dtype=np.int32
        )
    )


# ============================================================
# PREDICT DISTANCES
# ============================================================

def get_distances(
    model,
    pairs
):

    x1, x2, labels = load_pairs(
        pairs
    )

    print(
        f"Predicting {len(labels)} pairs..."
    )

    distances = model.predict(
        [x1, x2],
        batch_size=BATCH_SIZE,
        verbose=1
    ).flatten()

    return labels, distances


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    y_true,
    distances
):

    # Try many possible thresholds
    thresholds = np.linspace(
        distances.min(),
        distances.max(),
        500
    )

    best_threshold = thresholds[0]

    best_f1 = -1

    best_accuracy = 0


    for threshold in thresholds:

        # Lower distance = genuine
        predictions = (
            distances <= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        current_accuracy = accuracy_score(
            y_true,
            predictions
        )


        # Primary metric = F1
        # Accuracy used as tie-breaker
        if (
            current_f1 > best_f1
            or
            (
                current_f1 == best_f1
                and current_accuracy > best_accuracy
            )
        ):

            best_f1 = current_f1

            best_accuracy = current_accuracy

            best_threshold = threshold


    return (
        float(best_threshold),
        float(best_f1)
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    matrix
):

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "Confusion Matrix - Signature Verification"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Forged", "Genuine"]
    )

    plt.yticks(
        [0, 1],
        ["Forged", "Genuine"]
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )


    # Write numbers inside matrix
    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )


    plt.tight_layout()

    path = (
        OUTPUT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved to: {path}"
    )


# ============================================================
# SAVE ROC CURVE
# ============================================================

def save_roc_curve(
    y_true,
    distances
):

    # Smaller distance means more genuine.
    # Therefore use negative distance as score.
    scores = -distances

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_true,
            scores
        )
    )

    roc_auc = auc(
        false_positive_rate,
        true_positive_rate
    )


    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - Signature Verification"
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()


    path = (
        OUTPUT_DIR /
        "roc_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"ROC curve saved to: {path}"
    )

    return float(roc_auc)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("       SIGNATURE FORGERY DETECTION")
    print("              MODEL EVALUATION")
    print("=" * 60)


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )


    print(
        f"\nLoading model from:"
    )

    print(
        MODEL_PATH
    )


    model = create_siamese_model()

    model.load_weights(
        str(MODEL_PATH)
    )


    print(
        "\nModel loaded successfully!"
    )


    # ========================================================
    # CREATE SAME DATA SPLIT
    # ========================================================

    print(
        "\nCreating writer-independent test split..."
    )


    (
        train_pairs,
        validation_pairs,
        test_pairs
    ) = create_dataset_splits(
        pairs_per_class=30
    )


    print(
        "\nDataset information:"
    )

    print(
        f"Training pairs: {len(train_pairs)}"
    )

    print(
        f"Validation pairs: {len(validation_pairs)}"
    )

    print(
        f"Test pairs: {len(test_pairs)}"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n")
    print("=" * 60)
    print("          FINDING OPTIMAL THRESHOLD")
    print("=" * 60)


    validation_labels, validation_distances = (
        get_distances(
            model,
            validation_pairs
        )
    )


    threshold, validation_f1 = (
        find_best_threshold(
            validation_labels,
            validation_distances
        )
    )


    print(
        f"\nOptimal threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Validation F1-score: "
        f"{validation_f1:.4f}"
    )


    # ========================================================
    # TEST SET
    # ========================================================

    print("\n")
    print("=" * 60)
    print("              FINAL TEST")
    print("=" * 60)


    test_labels, test_distances = (
        get_distances(
            model,
            test_pairs
        )
    )


    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    test_predictions = (
        test_distances <= threshold
    ).astype(int)


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        test_labels,
        test_predictions
    )

    precision = precision_score(
        test_labels,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        test_labels,
        test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        test_labels,
        test_predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        test_labels,
        -test_distances
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("             FINAL MODEL RESULTS")
    print("=" * 60)


    print(
        f"\nAccuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"Threshold: {threshold:.6f}"
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        test_labels,
        test_predictions
    )


    print("\n")
    print("Confusion Matrix:")

    print(matrix)


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("Classification Report:")

    print(
        classification_report(
            test_labels,
            test_predictions,
            target_names=[
                "Forged",
                "Genuine"
            ],
            zero_division=0
        )
    )


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    save_confusion_matrix(
        matrix
    )


    # ========================================================
    # SAVE ROC CURVE
    # ========================================================

    save_roc_curve(
        test_labels,
        test_distances
    )


    # ========================================================
    # SAVE METRICS AS JSON
    # ========================================================

    results = {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc": float(roc_auc),

        "optimal_threshold": float(
            threshold
        ),

        "validation_f1": float(
            validation_f1
        ),

        "training_pairs": len(
            train_pairs
        ),

        "validation_pairs": len(
            validation_pairs
        ),

        "test_pairs": len(
            test_pairs
        )
    }


    metrics_path = (
        OUTPUT_DIR /
        "metrics.json"
    )


    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )


    print(
        f"\nMetrics saved to:"
    )

    print(
        metrics_path
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("             EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    main()