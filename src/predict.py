import numpy as np

from .preprocessing import preprocess_signature
from .model import create_siamese_model
from .config import MODEL_PATH


def load_model():

    model = create_siamese_model()

    model.load_weights(
        str(MODEL_PATH)
    )

    return model


def compare_signatures(
    reference_path,
    questioned_path
):

    model = load_model()

    reference = preprocess_signature(
        reference_path
    )

    questioned = preprocess_signature(
        questioned_path
    )

    reference = np.expand_dims(
        reference,
        axis=0
    )

    questioned = np.expand_dims(
        questioned,
        axis=0
    )

    distance = model.predict(
        [reference, questioned],
        verbose=0
    )[0][0]

    return float(distance)


def verify_signature(
    reference_path,
    questioned_path,
    threshold
):

    distance = compare_signatures(
        reference_path,
        questioned_path
    )

    if distance <= threshold:

        verdict = "GENUINE"

    else:

        verdict = "FORGED"

    return {
        "verdict": verdict,
        "distance": distance,
        "threshold": threshold
    }