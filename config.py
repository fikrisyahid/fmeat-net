import torch

# GENERAL CONFIGS
MODEL_USED = "CNN"  # MODEL_USED should be either "CNN" or "VGG"

# HYPERPARAMETER CONFIGS
DATASET_DIR = "./dataset/augmented"
EPOCH_AMOUNT = 10
DROPOUT_RATE = 0.8
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
CLASS_AMOUNT = 3

# CNN CONV LAYER CONFIGS
AMOUNT_OF_POOLING = 4
IMAGE_X_Y_SIZE = 112
LAST_CONV_LAYER_IMAGE_SIZE = (IMAGE_X_Y_SIZE // (2**AMOUNT_OF_POOLING)) ** 2

# MIXED PRECISION CONFIGS
MIXED_PRECISION_MODE = 1
MIXED_PRECISION_MODE_COLLECTION = {
    0: (torch.float32, torch.float32),
    1: (torch.float16, torch.float32),
    2: (torch.float64, torch.float64),  # Slower, cuDNN does not support float64
    # 3: (torch.float16, torch.float64), Not supported by autocast
    # 4: (torch.float32, torch.float64), Not supported by autocast
}
CONV_DTYPE, FC_DTYPE = MIXED_PRECISION_MODE_COLLECTION.get(
    MIXED_PRECISION_MODE, (torch.float32, torch.float32)
)

# LOG CONFIGS
PRINT_MEMORY_USAGE = False
SUMMARY_DISPLAYED = MIXED_PRECISION_MODE == 0 or MIXED_PRECISION_MODE == 1

TRAINING_COMBINATION = [
    {
        "model_type": "CNN",
        "pso": False,
        "mp_mode": 0,
    },
    {
        "model_type": "CNN",
        "pso": True,
        "mp_mode": 0,
    },
    {
        "model_type": "CNN",
        "pso": False,
        "mp_mode": 1,
    },
    {
        "model_type": "CNN",
        "pso": False,
        "mp_mode": 2,
    },
    {
        "model_type": "CNN",
        "pso": True,
        "mp_mode": 1,
    },
    {
        "model_type": "CNN",
        "pso": True,
        "mp_mode": 2,
    },
    {
        "model_type": "VGG",
        "pso": False,
        "mp_mode": 0,
    },
    {
        "model_type": "VGG",
        "pso": True,
        "mp_mode": 0,
    },
    {
        "model_type": "VGG",
        "pso": False,
        "mp_mode": 1,
    },
    {
        "model_type": "VGG",
        "pso": False,
        "mp_mode": 2,
    },
    {
        "model_type": "VGG",
        "pso": True,
        "mp_mode": 1,
    },
    {
        "model_type": "VGG",
        "pso": True,
        "mp_mode": 2,
    },
]
