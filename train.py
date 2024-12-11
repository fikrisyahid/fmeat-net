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
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pynvml
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0 (NVIDIA)


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
        training_time = end_time - start_time
        train_accuracy = correct_train / total_train
        train_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{config.EPOCH_AMOUNT}, Time: {training_time:.2f} seconds"
        )
        print(
            f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy:.4f}"
        )

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

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
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        precision = precision_score(
            all_labels, all_predictions, average="weighted"
        )
        recall = recall_score(all_labels, all_predictions, average="weighted")
        f1 = f1_score(all_labels, all_predictions, average="weighted")

        # Get GPU power usage
        power_usage = (
            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
        )

        # Get GPU memory usage
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_usage = memory_info.used / (1024**2)  # Convert to MB

        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n"
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, "
            f"GPU Power Usage: {power_usage:.2f} W, "
            f"GPU VRAM Usage: {vram_usage:.2f} MB\n"
        )

        # Ensure the log directory exists
        os.makedirs(config.LOG_FOLDER, exist_ok=True)

        # Define the log file path
        file_name = helper.generate_log_file_name(
            model_type, learning_rate, dropout_rate, batch_size, mp_mode
        )
        log_file_path = f"{config.LOG_FOLDER}/{file_name}.csv"

        # Create a DataFrame with the metrics
        metrics_df = pd.DataFrame(
            {
                "epoch": [epoch + 1],
                "training_loss": [train_loss],
                "training_accuracy": [train_accuracy],
                "validation_loss": [val_loss],
                "validation_accuracy": [val_accuracy],
                "precision": [precision],
                "recall": [recall],
                "f1_score": [f1],
                "gpu_watt_usage": [power_usage],
                "gpu_vram_usage": [vram_usage],
                "training_time": [training_time],
            }
        )

        # Append the metrics to the CSV file, creating it if it does not exist
        if not os.path.isfile(log_file_path):
            metrics_df.to_csv(log_file_path, index=False)
        else:
            metrics_df.to_csv(
                log_file_path, mode="a", header=False, index=False
            )

    # Testing phase
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_test_labels = []
    all_test_predictions = []

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
            all_test_labels.extend(labels.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total
    test_precision = precision_score(
        all_test_labels, all_test_predictions, average="weighted"
    )
    test_recall = recall_score(
        all_test_labels, all_test_predictions, average="weighted"
    )
    test_f1 = f1_score(
        all_test_labels, all_test_predictions, average="weighted"
    )

    print(
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
        f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}\n"
    )

    testing_log_file_path = (
        f"{config.LOG_FOLDER}/{config.TESTING_ACCURACY_LOG_FILE_NAME}"
    )

    # Create a DataFrame with the testing metrics
    test_metrics_df = pd.DataFrame(
        {
            "model": [model_type],
            "learning_rate": [learning_rate],
            "dropout_rate": [dropout_rate],
            "batch_size": [batch_size],
            "mixed_precision_mode": [mp_mode],
            "test_loss": [test_loss],
            "test_accuracy": [test_accuracy],
            "test_precision": [test_precision],
            "test_recall": [test_recall],
            "test_f1_score": [test_f1],
        }
    )

    # Append the testing metrics to the CSV file, creating it if it does not exist
    if not os.path.isfile(testing_log_file_path):
        test_metrics_df.to_csv(testing_log_file_path, index=False)
    else:
        test_metrics_df.to_csv(
            testing_log_file_path, mode="a", header=False, index=False
        )

    # Generate and save confusion matrix image
    conf_matrix = confusion_matrix(all_test_labels, all_test_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        xticklabels=["babi", "oplosan", "sapi"],
        yticklabels=["babi", "oplosan", "sapi"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    conf_matrix_image_path = (
        f"{config.LOG_FOLDER}/{file_name}_confusion_matrix.png"
    )
    plt.savefig(conf_matrix_image_path)
    plt.close()

    # Generate and save classification report
    class_report = classification_report(
        all_test_labels,
        all_test_predictions,
        target_names=["babi", "oplosan", "sapi"],
    )
    class_report_file_path = (
        f"{config.LOG_FOLDER}/{file_name}_classification_report.txt"
    )
    with open(class_report_file_path, "w") as f:
        f.write(class_report)

    # Save the trained model
    model_file_path = f"{config.LOG_FOLDER}/{file_name}_model.pth"
    torch.save(model.state_dict(), model_file_path)

    # Shutdown NVML
    pynvml.nvmlShutdown()
