import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run the model with specific configurations."
    )
    parser.add_argument(
        "model_type",
        type=lambda x: (str(x).lower()),
        help="Type of model to use: CNN or VGG16",
    )
    parser.add_argument(
        "learning_rate",
        type=float,
        help="Learning rate for model training",
    )
    parser.add_argument(
        "dropout_rate",
        type=float,
        help="Dropout rate for model training",
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="Batch size for model training",
    )
    parser.add_argument(
        "mp_mode", type=int, help="Mixed Precision Mode: 0, 1, or 2"
    )

    args = parser.parse_args()

    print(f"Model type: {args.model_type}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Dropout Rate: {args.dropout_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mixed Precision Mode: {args.mp_mode}")

    print("Running main function...")
    train(
        args.model_type,
        args.learning_rate,
        args.dropout_rate,
        args.batch_size,
        args.mp_mode,
    )
