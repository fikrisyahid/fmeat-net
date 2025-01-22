import torch

# =============================================================================
# Default hyperparameters
# =============================================================================
DATASET_DIR = "./dataset/augmented"
EPOCH_AMOUNT = 20
DROPOUT_RATE = 0.8
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
CLASS_AMOUNT = 3

# =============================================================================
# CNN conv layer settings
# =============================================================================
AMOUNT_OF_POOLING = 4
IMAGE_X_Y_SIZE = 112
LAST_CONV_LAYER_IMAGE_SIZE = (IMAGE_X_Y_SIZE // (2**AMOUNT_OF_POOLING)) ** 2

# =============================================================================
# Mixed precision settings
# =============================================================================
# MIXED_PRECISION_MODE = 1
MIXED_PRECISION_MODE_COLLECTION = {
    0: (torch.float32, torch.float32),
    1: (torch.float16, torch.float32),
    2: (torch.float64, torch.float64),  # Slower, cuDNN does not support float64
    # 3: (torch.float16, torch.float64), Not supported by autocast
    # 4: (torch.float32, torch.float64), Not supported by autocast
}

# =============================================================================
# Log settings
# =============================================================================
PRINT_MEMORY_USAGE = False
LOG_FOLDER = "./logs-augmented"
TESTING_ACCURACY_LOG_FILE_NAME = "testing_accuracy.csv"
ITERATION_LOG_FILE_NAME = "iteration_log.txt"  # Store only the filename
SAVE_MODEL = True
