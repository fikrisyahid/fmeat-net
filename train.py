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
from pyswarms.single import GlobalBestPSO


def main(model_type, pso, mp_mode):
    # Initialize mixed precision mode
    CONV_DTYPE, FC_DTYPE = config.MIXED_PRECISION_MODE_COLLECTION.get(mp_mode, (torch.float32, torch.float32))

    # Set summary displayed condition
    SUMMARY_DISPLAYED = mp_mode == 0 or mp_mode == 1

    # Initialize the model
    model = models.CNNModel() if model_type == "CNN" else models.VGGModel()
    image_size_based_on_model = 224 if model_type == "VGG" else config.IMAGE_X_Y_SIZE

    # Set the model to the desired dtype (all layers)
    model.to(FC_DTYPE)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Display model summary
    if SUMMARY_DISPLAYED:
        summary(
            model,
            input_size=(3, image_size_based_on_model, image_size_based_on_model),
        )

    # Define transformations for the training and validation sets
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size_based_on_model, image_size_based_on_model)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.5484, 0.3619, 0.3821] if model_type == "CNN" else [0.5480, 0.3614, 0.3816],
                std=[0.1129, 0.1049, 0.1092] if model_type == "CNN" else [0.1173, 0.1091, 0.1134],
            ),
        ]
    )

    # Load the dataset
    train_dataset = datasets.ImageFolder(os.path.join(config.DATASET_DIR, "training"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(config.DATASET_DIR, "validation"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(config.DATASET_DIR, "testing"), transform=transform)

    # Make the loader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Define training function
    def train_model(x):
        using_pso = x is not None

        # Ambil hyperparameter dari partikel PSO
        learning_rate = x[:, 0] if using_pso else 0  # Dimensi 0: learning rate
        batch_size = x[:, 1] if using_pso else 0  # Dimensi 1: batch size
        dropout_rate = x[:, 2] if using_pso else 0  # Dimensi 2: dropout rate

        # Simulasi perhitungan "loss" dari model yang pakai hyperparameter ini
        # Misalnya kita buat loss = (learning_rate - 0.005)^2 + (batch_size - 64)^2 + (dropout_rate - 0.2)^2
        # Tujuannya biar nilai loss sekecil mungkin di sekitar learning rate = 0.005, batch size = 64, dropout rate = 0.2
        loss = (learning_rate - 0.005) ** 2 + (batch_size - 64) ** 2 + (dropout_rate - 0.2) ** 2

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
                    outputs = model(inputs, epoch)
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

            print(f"Epoch {epoch + 1}/{config.EPOCH_AMOUNT}, Time: {iteration_time:.2f} seconds")
            print(f"Training Loss: {running_loss / len(train_loader)}, Training Accuracy: {train_accuracy:.4f}")

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
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}\n")

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

        return -test_accuracy

    if pso:
        # Set bounds untuk tiap hyperparameter
        bounds = (
            [
                0.001,  # Lower bound untuk learning rate
                16,  # Lower bound untuk batch size
                0.1,  # Lower bound untuk dropout rate
            ],
            [
                0.01,  # Upper bound untuk learning rate
                128,  # Upper bound untuk batch size
                0.5,  # Upper bound untuk dropout rate
            ],
        )

        # Set options untuk PSO
        options = {
            "c1": 1.5,  # Cognitive parameter
            "c2": 2.0,  # Social parameter
            "w": 0.5,  # Inertia weight
        }

        # Inisialisasi PSO optimizer
        optimizer = GlobalBestPSO(n_particles=40, dimensions=3, options=options, bounds=bounds)

        # Run optimization
        best_score, best_params = optimizer.optimize(train_model, iters=100)

        # Print hasil optimasi
        print("Best Score (minimized loss):", best_score)
        print("Best Hyperparameters:")
        print("Learning Rate:", best_params[0])
        print("Batch Size:", int(best_params[1]))
        print("Dropout Rate:", best_params[2])
    else:
        train_model(None)
