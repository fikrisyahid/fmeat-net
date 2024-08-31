import torch
import config


def print_current_memory_usage(layer_str, before=False):
    if config.PRINT_MEMORY_USAGE:
        print(
            f"Memory usage {"before" if before else "after"} {layer_str}:",
            torch.cuda.memory_allocated() / 1024**3,
            "GB",
        )
