import tensorflow as tf
from tensorflow.keras import layers, Model

from .config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    CHANNELS,
    EMBEDDING_DIM,
    MARGIN,
    LEARNING_RATE
)


# ============================================================
# 1. CNN EMBEDDING NETWORK
# ============================================================

def create_embedding_network():

    inputs = layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            CHANNELS
        ),
        name="signature_input"
    )

    # --------------------------------------------------------
    # Convolution Block 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # --------------------------------------------------------
    # Convolution Block 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # --------------------------------------------------------
    # Convolution Block 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)


    # --------------------------------------------------------
    # Convolution Block 4
    # --------------------------------------------------------

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)


    # --------------------------------------------------------
    # Global Average Pooling
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D()(x)


    # --------------------------------------------------------
    # Fully Connected Layer
    # --------------------------------------------------------

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.3
    )(x)


    # --------------------------------------------------------
    # Signature Embedding
    # --------------------------------------------------------

    outputs = layers.Dense(
        EMBEDDING_DIM,
        name="signature_embedding"
    )(x)


    # Normalize embedding
    outputs = layers.Lambda(
        lambda x: tf.math.l2_normalize(
            x,
            axis=1
        ),
        name="normalized_embedding"
    )(outputs)


    return Model(
        inputs,
        outputs,
        name="signature_embedding_network"
    )


# ============================================================
# 2. EUCLIDEAN DISTANCE
# ============================================================

def euclidean_distance(vectors):

    embedding_a, embedding_b = vectors

    squared_difference = tf.square(
        embedding_a - embedding_b
    )

    sum_squared_difference = tf.reduce_sum(
        squared_difference,
        axis=1,
        keepdims=True
    )

    distance = tf.sqrt(
        tf.maximum(
            sum_squared_difference,
            tf.keras.backend.epsilon()
        )
    )

    return distance


# ============================================================
# 3. CONTRASTIVE LOSS
# ============================================================

def contrastive_loss(y_true, distance):

    y_true = tf.cast(
        tf.reshape(
            y_true,
            [-1]
        ),
        distance.dtype
    )

    distance = tf.reshape(
        distance,
        [-1]
    )

    # Similar pairs
    positive_loss = (
        y_true *
        tf.square(distance)
    )

    # Different pairs
    negative_loss = (
        (1 - y_true) *
        tf.square(
            tf.maximum(
                MARGIN - distance,
                0
            )
        )
    )

    return tf.reduce_mean(
        positive_loss + negative_loss
    )


# ============================================================
# 4. CREATE SIAMESE NETWORK
# ============================================================

def create_siamese_model():

    # Shared CNN
    embedding_network = (
        create_embedding_network()
    )


    # First signature
    input_a = layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            CHANNELS
        ),
        name="signature_a"
    )


    # Second signature
    input_b = layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            CHANNELS
        ),
        name="signature_b"
    )


    # Generate embeddings
    embedding_a = embedding_network(
        input_a
    )

    embedding_b = embedding_network(
        input_b
    )


    # Calculate distance
    distance = layers.Lambda(
        euclidean_distance,
        name="euclidean_distance"
    )(
        [embedding_a, embedding_b]
    )


    # Complete Siamese model
    model = Model(
        inputs=[
            input_a,
            input_b
        ],
        outputs=distance,
        name="siamese_signature_verification"
    )


    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=contrastive_loss
    )


    return model