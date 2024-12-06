import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import os
import time
from tqdm import tqdm
import models
import config


def train(model_type, learning_rate, dropout_rate, batch_size, mp_mode):
    # Initialize mixed precision mode
    CONV_DTYPE, FC_DTYPE = config.MIXED_PRECISION_MODE_COLLECTION.get(
        mp_mode, (torch.float32, torch.float32)
    )

    # Set summary displayed condition
    SUMMARY_DISPLAYED = mp_mode == 0 or mp_mode == 1

    # Initialize the model
    model = (
        models.CNNModel(dropout_rate)
        if model_type == "cnn"
        else models.VGGModel(dropout_rate)
    )
    image_size_based_on_model = (
        224 if model_type == "vgg" else config.IMAGE_X_Y_SIZE
    )

    # Set the model to the desired dtype (all layers)
    model.to(FC_DTYPE)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Display model summaryi
    if SUMMARY_DISPLAYED:
        summary(
            model,
            input_size=(
                3,
                image_size_based_on_model,
                image_size_based_on_model,
            ),
        )

    # Define transformations for the training and validation sets
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size_based_on_model, image_size_based_on_model)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.5484, 0.3619, 0.3821]
                if model_type == "cnn"
                else [0.5480, 0.3614, 0.3816],
                std=[0.1129, 0.1049, 0.1092]
                if model_type == "cnn"
                else [0.1173, 0.1091, 0.1134],
            ),
        ]
    )

    # Load the dataset
    train_dataset = datasets.ImageFolder(
        os.path.join(config.DATASET_DIR, "training"), transform=transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(config.DATASET_DIR, "validation"), transform=transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(config.DATASET_DIR, "testing"), transform=transform
    )

    # Make the loader for each dataset
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in tqdm(range(config.EPOCH_AMOUNT), desc="Epochs"):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time = time.time()
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            inputs = inputs.to(FC_DTYPE)
            with torch.autocast(device_type="cuda", dtype=CONV_DTYPE):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        end_time = time.time()
        iteration_time = end_time - start_time
        train_accuracy = correct_train / total_train

        print(
            f"Epoch {epoch + 1}/{config.EPOCH_AMOUNT}, Time: {iteration_time:.2f} seconds"
        )
        print(
            f"Training Loss: {running_loss / len(train_loader)}, Training Accuracy: {train_accuracy:.4f}"
        )

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.to(FC_DTYPE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        print(
            f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}\n"
        )

    # Testing phase
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(FC_DTYPE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
