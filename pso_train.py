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
import helper
from pyswarms.single import GlobalBestPSO
import numpy as np


def main(model_type, pso, mp_mode):
    # Define training function
    def train_model(x):
        using_pso = x is not None

        testing_accuracies = []

        for particle_index in range(len(x) if using_pso else 1):
            if using_pso:
                print(f"Current particle index: {particle_index}")
                print(f"Current hyperparameters: {x[particle_index]}")

            # Define hyperparameters
            learning_rate = (
                x[:, 0][particle_index] if using_pso else config.LEARNING_RATE
            )
            batch_size = (
                helper.round_to_closest_possible(
                    config.PSO_BATCH_SIZE, x[:, 1][particle_index]
                )
                if using_pso
                else config.BATCH_SIZE
            )
            fine_tune_unfreeze_layer = (
                helper.round_to_closest_possible(
                    config.PSO_FINE_TUNE_UNFREEZE_LAYER, x[:, 2][particle_index]
                )
                if using_pso
                else config.FINE_TUNE_UNFREEZE_LAYER
            )
            dropout_rate = (
                x[:, 3][particle_index] if using_pso else config.DROPOUT_RATE
            )

            if using_pso:
                print(f"Final learning rate: {learning_rate}")
                print(f"Final batch size: {batch_size}")
                print(
                    f"Final fine tune unfreeze layer: {fine_tune_unfreeze_layer}"
                )
                print(f"Final dropout rate: {dropout_rate}")

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
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
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
                    v2.Resize(
                        (image_size_based_on_model, image_size_based_on_model)
                    ),
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
                os.path.join(config.DATASET_DIR, "training"),
                transform=transform,
            )
            val_dataset = datasets.ImageFolder(
                os.path.join(config.DATASET_DIR, "validation"),
                transform=transform,
            )
            test_dataset = datasets.ImageFolder(
                os.path.join(config.DATASET_DIR, "testing"), transform=transform
            )

            # Make the loader for each dataset
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

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
                        outputs = model(
                            inputs,
                            fine_tune_unfreeze_layer,
                            epoch,
                        )
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

            testing_accuracies.append(
                -test_accuracy if using_pso else test_accuracy
            )

        return testing_accuracies if using_pso else testing_accuracies[0]

    if pso:
        # Set bounds untuk tiap hyperparameter
        bounds = (
            [
                0.0001
                if model_type == "cnn"
                else 0.000001,  # Lower bound untuk learning rate
                16,  # Lower bound untuk batch size
                0,  # Lower bound untuk fine tune layer
                0.1,  # Lower bound untuk dropout rate
            ],
            [
                0.1
                if model_type == "cnn"
                else 0.001,  # Upper bound untuk learning rate
                256,  # Upper bound untuk batch size
                5,  # Upper bound untuk fine tune layer
                0.8,  # Upper bound untuk dropout rate
            ],
        )

        # Set options untuk PSO
        options = {
            "c1": 1.5,  # Cognitive parameter
            "c2": 2.0,  # Social parameter
            "w": 0.5,  # Inertia weight
        }

        # Inisialisasi PSO optimizer
        optimizer = GlobalBestPSO(
            n_particles=5,
            dimensions=4,
            options=options,
            bounds=bounds,
            init_pos=np.array(
                [
                    [0.0001, 16, 0, 0.5],
                    [0.001, 32, 1, 0.2],
                    [0.005, 64, 2, 0.3],
                    [0.007, 128, 3, 0.6],
                    [0.0007, 200, 4, 0.7],
                ]
            ),
        )

        # Run optimization
        best_score, best_params = optimizer.optimize(train_model, iters=10)

        # Print hasil optimasi
        print("Best Score (minimized loss):", best_score)
        print("Best Hyperparameters:")
        print("Learning Rate:", best_params[0])
        print("Batch Size:", int(best_params[1]))
        print("Dropout Rate:", best_params[2])
    else:
        train_model(None)
