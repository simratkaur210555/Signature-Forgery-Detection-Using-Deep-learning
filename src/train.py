import random
import numpy as np
import tensorflow as tf

from .dataset import create_dataset_splits
from .preprocessing import preprocess_signature
from .model import create_siamese_model

from .config import (
    BATCH_SIZE,
    EPOCHS,
    MODEL_PATH,
    RANDOM_STATE
)


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# PREPROCESSED IMAGE CACHE
# ============================================================

IMAGE_CACHE = {}


def get_cached_image(path):

    path_string = str(path)

    if path_string not in IMAGE_CACHE:

        IMAGE_CACHE[path_string] = (
            preprocess_signature(path)
        )

    return IMAGE_CACHE[path_string]


# ============================================================
# DATA GENERATOR
# ============================================================

class SignaturePairSequence(
    tf.keras.utils.Sequence
):

    def __init__(
        self,
        pairs,
        batch_size=BATCH_SIZE,
        shuffle=True
    ):

        self.pairs = pairs

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indices = np.arange(
            len(self.pairs)
        )

        self.on_epoch_end()


    def __len__(self):

        return int(
            np.ceil(
                len(self.pairs) /
                self.batch_size
            )
        )


    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        images_a = []
        images_b = []
        labels = []


        for i in batch_indices:

            path_a, path_b, label = (
                self.pairs[i]
            )

            images_a.append(
                get_cached_image(path_a)
            )

            images_b.append(
                get_cached_image(path_b)
            )

            labels.append(label)


        return (
            (
                np.asarray(
                    images_a,
                    dtype=np.float32
                ),

                np.asarray(
                    images_b,
                    dtype=np.float32
                )
            ),

            np.asarray(
                labels,
                dtype=np.float32
            )
        )


    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("     FAST SIGNATURE VERIFICATION TRAINING")
    print("=" * 60)


    # --------------------------------------------------------
    # Create balanced, sampled dataset
    # --------------------------------------------------------

    print("\nCreating optimized dataset...")

    (
        train_pairs,
        validation_pairs,
        test_pairs
    ) = create_dataset_splits(
        pairs_per_class=30
    )


    # --------------------------------------------------------
    # Generators
    # --------------------------------------------------------

    print("\nCreating generators...")

    train_generator = SignaturePairSequence(
        train_pairs,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_generator = SignaturePairSequence(
        validation_pairs,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    print(
        f"\nTraining batches: "
        f"{len(train_generator)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_generator)}"
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nBuilding Siamese CNN...")

    model = create_siamese_model()

    print("\nModel ready!")


    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("              STARTING TRAINING")
    print("=" * 60)


    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model.save(
        str(MODEL_PATH)
    )


    print("\n")
    print("=" * 60)
    print("             TRAINING COMPLETED")
    print("=" * 60)


    print(
        f"\nModel saved to:"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":

    main()