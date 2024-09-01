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


# Initialize the model
model = models.CNNModel() if config.MODEL_USED == "CNN" else models.VGGModel()
image_size_based_on_model = (
    224 if config.MODEL_USED == "VGG" else config.IMAGE_X_Y_SIZE
)

# Set the model to the desired dtype (all layers)
model.to(config.FC_DTYPE)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Display model summary
if config.SUMMARY_DISPLAYED:
    summary(
        model,
        input_size=(3, image_size_based_on_model, image_size_based_on_model),
    )

# Define transformations for the training and validation sets
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((image_size_based_on_model, image_size_based_on_model)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(10),
        v2.RandomPerspective(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# Load the dataset
train_dataset = datasets.ImageFolder(
    os.path.join(config.DATASET_DIR, "training"), transform=transform
)
val_dataset = datasets.ImageFolder(
    os.path.join(config.DATASET_DIR, "validation"), transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# Training loop
for epoch in tqdm(range(config.EPOCH_AMOUNT), desc="Epochs"):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        inputs = inputs.to(config.FC_DTYPE)
        with torch.autocast(device_type="cuda", dtype=config.CONV_DTYPE):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    end_time = time.time()
    iteration_time = end_time - start_time

    print(
        f"Epoch {epoch + 1}/{config.EPOCH_AMOUNT}, Loss: {running_loss / len(train_loader)}, Time: {iteration_time:.2f} seconds"
    )

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(config.FC_DTYPE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
