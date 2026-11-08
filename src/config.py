from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "CEDAR"

GENUINE_DIR = DATASET_DIR / "full_org"
FORGED_DIR = DATASET_DIR / "full_forg"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "signature_siamese.keras"


# ==========================================
# IMAGE SETTINGS
# ==========================================

IMG_HEIGHT = 155
IMG_WIDTH = 220
CHANNELS = 1


# ==========================================
# TRAINING SETTINGS
# ==========================================

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.0001

EMBEDDING_DIM = 128

MARGIN = 1.0

RANDOM_STATE = 42