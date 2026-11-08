from pathlib import Path
import random
import re

from .config import GENUINE_DIR, FORGED_DIR, RANDOM_STATE


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
}


def get_image_files(directory):
    directory = Path(directory)

    return sorted([
        file
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ])


def extract_writer_id(filename):
    """
    CEDAR format:

    original_1_1.png
    forgeries_1_1.png

    Writer ID = 1
    """

    filename = Path(filename).name

    match = re.match(
        r"(?:original|forgeries)_(\d+)_(\d+)",
        filename,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


def extract_signature_number(filename):

    filename = Path(filename).name

    match = re.match(
        r"(?:original|forgeries)_(\d+)_(\d+)",
        filename,
        re.IGNORECASE
    )

    if match:
        return int(match.group(2))

    return None


def build_signature_dictionary():

    genuine_files = get_image_files(
        GENUINE_DIR
    )

    forged_files = get_image_files(
        FORGED_DIR
    )

    genuine = {}
    forged = {}

    for file in genuine_files:

        writer_id = extract_writer_id(file)

        if writer_id is not None:
            genuine.setdefault(
                writer_id,
                []
            ).append(file)

    for file in forged_files:

        writer_id = extract_writer_id(file)

        if writer_id is not None:
            forged.setdefault(
                writer_id,
                []
            ).append(file)

    return genuine, forged


def create_pairs_for_writers(
    writer_ids,
    genuine,
    forged,
    pairs_per_class=30
):

    random.seed(RANDOM_STATE)

    pairs = []

    for writer_id in sorted(writer_ids):

        genuine_signatures = genuine[writer_id]

        forged_signatures = forged.get(
            writer_id,
            []
        )

        # ------------------------------------------------
        # Genuine-Genuine pairs
        # Label = 1
        # ------------------------------------------------

        genuine_pairs = []

        for i in range(
            len(genuine_signatures)
        ):

            for j in range(
                i + 1,
                len(genuine_signatures)
            ):

                genuine_pairs.append(
                    (
                        genuine_signatures[i],
                        genuine_signatures[j],
                        1
                    )
                )

        # Randomly sample genuine pairs
        random.shuffle(genuine_pairs)

        genuine_pairs = genuine_pairs[
            :pairs_per_class
        ]

        pairs.extend(genuine_pairs)

        # ------------------------------------------------
        # Genuine-Forged pairs
        # Label = 0
        # ------------------------------------------------

        forged_pairs = []

        for genuine_path in genuine_signatures:

            for forged_path in forged_signatures:

                forged_pairs.append(
                    (
                        genuine_path,
                        forged_path,
                        0
                    )
                )

        random.shuffle(forged_pairs)

        forged_pairs = forged_pairs[
            :pairs_per_class
        ]

        pairs.extend(forged_pairs)

    random.shuffle(pairs)

    return pairs


def create_dataset_splits(
    pairs_per_class=30
):

    genuine, forged = (
        build_signature_dictionary()
    )

    writer_ids = sorted(genuine.keys())

    print(
        f"Total writers: {len(writer_ids)}"
    )

    # -----------------------------------------------
    # 80% train
    # 10% validation
    # 10% test
    # -----------------------------------------------

    random.seed(RANDOM_STATE)

    shuffled_writers = writer_ids.copy()

    random.shuffle(
        shuffled_writers
    )

    train_count = int(
        len(shuffled_writers) * 0.80
    )

    validation_count = int(
        len(shuffled_writers) * 0.10
    )

    train_writers = shuffled_writers[
        :train_count
    ]

    validation_writers = shuffled_writers[
        train_count:
        train_count + validation_count
    ]

    test_writers = shuffled_writers[
        train_count + validation_count:
    ]

    # -----------------------------------------------
    # Generate pairs
    # -----------------------------------------------

    train_pairs = create_pairs_for_writers(
        train_writers,
        genuine,
        forged,
        pairs_per_class
    )

    validation_pairs = create_pairs_for_writers(
        validation_writers,
        genuine,
        forged,
        pairs_per_class
    )

    test_pairs = create_pairs_for_writers(
        test_writers,
        genuine,
        forged,
        pairs_per_class
    )

    print("\nWriter split:")
    print(
        f"Training writers: {train_writers}"
    )

    print(
        f"Validation writers: {validation_writers}"
    )

    print(
        f"Testing writers: {test_writers}"
    )

    print("\nPair split:")
    print(
        f"Training pairs: {len(train_pairs)}"
    )

    print(
        f"Validation pairs: {len(validation_pairs)}"
    )

    print(
        f"Testing pairs: {len(test_pairs)}"
    )

    return (
        train_pairs,
        validation_pairs,
        test_pairs
    )


# ------------------------------------------------------------
# Compatibility function
# ------------------------------------------------------------

def create_pairs():

    genuine, forged = (
        build_signature_dictionary()
    )

    return create_pairs_for_writers(
        genuine.keys(),
        genuine,
        forged,
        pairs_per_class=30
    )


if __name__ == "__main__":

    (
        train_pairs,
        validation_pairs,
        test_pairs
    ) = create_dataset_splits()

    print("\nDataset ready!")