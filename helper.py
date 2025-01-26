import torch
import config
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import itertools
import pandas as pd
import seaborn as sns


def generate_log_file_name(
    model_type, learning_rate, dropout_rate, batch_size, mp_mode
):
    # Generate log file name based on model type and hyperparameters
    log_file_name = f"{model_type}_lr{learning_rate}_dr{dropout_rate}_bs{batch_size}_mp{mp_mode}"
    return log_file_name


def print_current_memory_usage(layer_str, before=False):
    if config.PRINT_MEMORY_USAGE:
        print(
            f"Memory usage {'before' if before else 'after'} {layer_str}:",
            torch.cuda.memory_allocated() / 1024**3,
            "GB",
        )


def get_normalization_mean_std(dataset_dir="./dataset/augmented/training"):
    # Define a transform that only converts to tensor (not yet normalized)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((112, 112), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # Load dataset
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize variables to store total mean and std
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.size(0)

    mean /= total_images
    std /= total_images

    return mean, std


def add_dark_red_tone(image):
    image_np = np.array(image)
    image_np[:, :, 0] = np.clip(image_np[:, :, 0] + 45, 0, 255)  # Red
    image_np[:, :, 1] = np.clip(image_np[:, :, 1] + 10, 0, 255)  # Green
    image_np[:, :, 2] = np.clip(image_np[:, :, 2] - 24, 0, 255)  # Blue
    # Reduce brightness for a darker effect
    image_np = np.clip(image_np * 0.5, 0, 255)
    return Image.fromarray(image_np.astype(np.uint8))


# Function to add a yellow bright tone
def add_yellow_tone(image):
    image_np = np.array(image)
    # Add to red and green channels to make it more yellowish
    image_np[:, :, 0] = np.clip(image_np[:, :, 0] + 30, 0, 255)  # Red
    image_np[:, :, 1] = np.clip(image_np[:, :, 1] + 40, 0, 255)  # Green
    return Image.fromarray(image_np.astype(np.uint8))


def generate_augmented_images(
    source_dir="./dataset/validation",
    destination_dir="./dataset/augmented/validation",
):
    # Define separate transforms for each augmentation type
    augmentation_transforms = {
        "resize": v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "flip_h": v2.Compose(
            [
                v2.RandomHorizontalFlip(p=1.0),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "flip_v": v2.Compose(
            [
                v2.RandomVerticalFlip(p=1.0),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "rotate": v2.Compose(
            [
                v2.RandomRotation(10),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "perspective": v2.Compose(
            [
                v2.RandomPerspective(p=1.0),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "grayscale": v2.Compose(
            [
                v2.Grayscale(num_output_channels=3),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "dark_red_tone": v2.Compose(
            [
                v2.Lambda(lambda img: add_dark_red_tone(img)),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "yellow_bright_tone": v2.Compose(
            [
                v2.Lambda(lambda img: add_yellow_tone(img)),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
    }

    augmented_dir = destination_dir
    os.makedirs(augmented_dir, exist_ok=True)

    # Generate augmented images for each transform type
    for aug_type, transform in augmentation_transforms.items():
        print(f"\nGenerating {aug_type} augmented images...")

        # Create dataset with current transform
        augmented_dataset = datasets.ImageFolder(
            root=source_dir, transform=transform
        )
        batch_size = 32
        train_loader = DataLoader(
            augmented_dataset, batch_size=batch_size, shuffle=False
        )

        # Generate and save augmented images
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")

            for i, (image, label) in enumerate(zip(images, labels)):
                class_name = augmented_dataset.classes[label]
                class_dir = os.path.join(augmented_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                save_path = os.path.join(
                    class_dir,
                    f"{class_name}_{aug_type}_{batch_idx * batch_size + i}.jpg",
                )
                save_image(image, save_path)
                print(f"Saved: {save_path}")

        print(f"Finished generating {aug_type} augmented images")


def visualize_augmentations(image_path):
    """
    Visualize original image and its augmented versions
    Args:
        image_path (str): Path to the input image
    """

    # Define transforms as before
    augmentation_transforms = {
        "Original": v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "Horizontal Flip": v2.Compose(
            [
                v2.RandomHorizontalFlip(p=1.0),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "Vertical Flip": v2.Compose(
            [
                v2.RandomVerticalFlip(p=1.0),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "Rotation": v2.Compose(
            [
                v2.RandomRotation(10),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "Perspective": v2.Compose(
            [
                v2.RandomPerspective(p=1.0),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "Grayscale": v2.Compose(
            [
                v2.Grayscale(num_output_channels=3),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "dark_red_tone": v2.Compose(
            [
                v2.Lambda(lambda img: add_dark_red_tone(img)),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "yellow_bright_tone": v2.Compose(
            [
                v2.Lambda(lambda img: add_yellow_tone(img)),
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
    }

    # Load image
    image = Image.open(image_path)

    # Create subplot grid with 2x4 layout instead of 2x3
    _, axes = plt.subplots(2, 4, figsize=(20, 10))  # Increased figure width
    axes = axes.ravel()

    # Generate and display augmentations
    for idx, (name, transform) in enumerate(augmentation_transforms.items()):
        # Apply transform
        aug_image = transform(image)

        # Display image
        axes[idx].imshow(aug_image.permute(1, 2, 0))
        axes[idx].set_title(name)
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def get_average_data_from_csv(new_column, calculated_key):
    # Define the parameter ranges
    model_types = ["cnn", "vgg"]
    learning_rates = [0.01, 0.001, 0.0001]
    dropout_rates = [0.2, 0.5, 0.8]
    batch_sizes = [16, 32, 64]
    mp_modes = [0, 1, 2]  # Mixed precision modes

    # Generate all combinations of parameters
    combinations = list(
        itertools.product(
            model_types, learning_rates, dropout_rates, batch_sizes, mp_modes
        )
    )

    log_dir = "./logs/non-augmented"

    # Create a new DataFrame with the required columns
    rows = []

    for (
        model_type,
        learning_rate,
        dropout_rate,
        batch_size,
        mp_mode,
    ) in combinations:
        log_file_name = generate_log_file_name(
            model_type, learning_rate, dropout_rate, batch_size, mp_mode
        )
        log_file_path = os.path.join(log_dir, f"{log_file_name}.csv")
        if os.path.exists(log_file_path):
            data = pd.read_csv(log_file_path)
            average_value = data[calculated_key].mean()
            print(f"{log_file_name} with {new_column}: {average_value} GB")

            # Append the row to the list
            rows.append(
                {
                    "model": model_type,
                    "learning_rate": learning_rate,
                    "dropout_rate": dropout_rate,
                    "batch_size": batch_size,
                    "mixed_precision_mode": mp_mode,
                    f"{new_column}": average_value,
                }
            )

    # After all iterations, create a DataFrame from the rows
    df = pd.DataFrame(rows)

    # Define the output file path
    output_file_path = os.path.join(log_dir, f"{new_column}.xlsx")

    # Check if the file already exists
    if os.path.exists(output_file_path):
        # If it exists, append the new data to the existing file
        existing_df = pd.read_excel(output_file_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    # Save the DataFrame to an Excel file
    df.to_excel(output_file_path, index=False)

    print("Finished generating VRAM usage CSV file.")


def fix_csv_combination_sort(csv_source_path, csv_destination_path):
    # Load the CSV file
    df = pd.read_csv(csv_source_path)

    # Sort the DataFrame based on the model type and hyperparameters
    df = df.sort_values(
        by=[
            "model",
            "learning_rate",
            "dropout_rate",
            "batch_size",
            "mixed_precision_mode",
        ],
        ascending=[
            True,
            False,
            True,
            True,
            True,
        ],  # Sort learning_rate in descending order
    )

    # Save the sorted DataFrame to a new CSV file
    df.to_csv(csv_destination_path, index=False)

    print("Finished sorting the CSV file.")


def get_correlation_matrix(excel_path):
    # Load the CSV file
    df = pd.read_excel(excel_path)

    # Filter string column
    df = df.select_dtypes(include=["float64", "int64"])

    # Get the correlation matrix
    corr = df.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    plt.matshow(corr, cmap="coolwarm", fignum=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

    print("Finished generating the correlation matrix.")


def ax_add_center_labels(ax):
    """
    ax: axes object dari matplotlib/seaborn
    fmt: format string untuk nilai label
    fontcolor: warna teks label
    """
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            color="black",
            fontsize=10,  # atur sesuai selera
        )


def plot_bar_mean(
    excel_path,
    groupby_column,
    y_column,
    x_column,
    x_label,
    y_label,
    hue_column,
    plot_title,
):
    df = pd.read_excel(excel_path)

    correlation_data = df.groupby(groupby_column)[y_column].mean().reset_index()

    print(correlation_data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=correlation_data,
        x=x_column,
        y=y_column,
        hue=hue_column,
        errorbar=None,
    )
    ax.set(xlabel=x_label, ylabel=y_label)
    plt.title(plot_title)
    ax_add_center_labels(ax)
    plt.show()
