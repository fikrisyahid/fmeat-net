import torch

EPOCH_AMOUNT = 1
IMAGE_X_Y_SIZE = 112
BATCH_SIZE = 32
AMOUNT_OF_POOLING = 3
LAST_CONV_LAYER_IMAGE_SIZE = (IMAGE_X_Y_SIZE // (2**AMOUNT_OF_POOLING)) ** 2
MODEL_USED = "CNN"  # MODEL_USED should be either "CNN" or "VGG"
PRINT_MEMORY_USAGE = True
MIXED_PRECISION_MODE = 1
MIXED_PRECISION_MODE_COLLECTION = {
    0: (torch.float32, torch.float32),
    1: (torch.float16, torch.float32),
    2: (torch.float64, torch.float64),  # Slower, cuDNN does not support float64
    # 3: (torch.float16, torch.float64), Not supported by autocast, same as number 2
    # 4: (torch.float32, torch.float64), Not supported by autocast, same as number 2
}

CONV_DTYPE, FC_DTYPE = MIXED_PRECISION_MODE_COLLECTION.get(
    MIXED_PRECISION_MODE, (torch.float32, torch.float32)
)
CLASS_AMOUNT = 3
DATASET_DIR = "cat_dog"
SUMMARY_DISPLAYED = MIXED_PRECISION_MODE == 0 or MIXED_PRECISION_MODE == 1
