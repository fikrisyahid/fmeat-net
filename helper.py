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


def print_current_memory_usage(layer_str, before=False):
    if config.PRINT_MEMORY_USAGE:
        print(
            f"Memory usage {"before" if before else "after"} {layer_str}:",
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
    # Convert to float for more precise manipulation
    image_np = image_np.astype(np.float32)
    # Increase red channel to get a deeper reddish tone
    image_np[:, :, 0] = np.clip(
        image_np[:, :, 0] + 40, 0, 255
    )  # Increase red significantly
    # Adjust green channel moderately to balance the tone
    image_np[:, :, 1] = np.clip(
        image_np[:, :, 1] + 20, 0, 255
    )  # Increase green slightly
    # Reduce blue channel to warm up the tone
    image_np[:, :, 2] = np.clip(
        image_np[:, :, 2] - 15, 0, 255
    )  # Decrease blue for warmth
    # Darken the image overall
    image_np = np.clip(image_np * 0.65, 0, 255)  # Reduce brightness significantly
    # Convert back to uint8
    image_np = image_np.astype(np.uint8)
    return Image.fromarray(image_np)


# Function to add a yellow bright tone
def add_yellow_tone(image):
    image_np = np.array(image)
    # Add to red and green channels to make it more yellowish
    image_np[:, :, 0] = np.clip(image_np[:, :, 0] + 30, 0, 255)  # Red
    image_np[:, :, 1] = np.clip(image_np[:, :, 1] + 40, 0, 255)  # Green
    return Image.fromarray(image_np.astype(np.uint8))


def generate_augmented_images(
    source_dir="./dataset/validation", destination_dir="./dataset/augmented/validation"
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
        # "Color_jitter": v2.Compose(
        #     [
        #         v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #         v2.ToImage(),
        #         v2.Resize((224, 224)),
        #         v2.ToDtype(torch.float32, scale=True),
        #     ]
        # ),
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
        augmented_dataset = datasets.ImageFolder(root=source_dir, transform=transform)
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
        # "Color_jitter": v2.Compose(
        #     [
        #         v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #         v2.ToImage(),
        #         v2.Resize((224, 224)),
        #         v2.ToDtype(torch.float32, scale=True),
        #     ]
        # ),
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
