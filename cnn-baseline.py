import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2
import os
import time
from tqdm import tqdm

###################################################################################
# Config                                                                          #
###################################################################################
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
    2: (torch.float64, torch.float64), # Slower, cuDNN does not support float64
    # 3: (torch.float16, torch.float64), Not supported by autocast, same as number 2
    # 4: (torch.float32, torch.float64), Not supported by autocast, same as number 2
}

CONV_DTYPE, FC_DTYPE = MIXED_PRECISION_MODE_COLLECTION.get(
    MIXED_PRECISION_MODE, (torch.float32, torch.float32)
)
CLASS_AMOUNT = 3
DATASET_DIR = "cat_dog"
SUMMARY_DISPLAYED = MIXED_PRECISION_MODE == 0 or MIXED_PRECISION_MODE == 1
###################################################################################

def print_current_memory_usage(layer_str, before=False):
    if PRINT_MEMORY_USAGE:
        print(
            f"Memory usage {"before" if before else "after"} {layer_str}:",
            torch.cuda.memory_allocated() / 1024**3,
            "GB",
        )


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        # Second layer
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        # Third layer
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # Fourth layer
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        # Fifth layer
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()

        # Sixth layer
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()

        # Seventh layer
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(256 * LAST_CONV_LAYER_IMAGE_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, CLASS_AMOUNT)  # Output 3 classes

        self.dropout = nn.Dropout(0.5)

    def forward_conv(self, x):
        print_current_memory_usage("conv1", before=True)
        x = self.relu1(self.bn1(self.conv1(x)))
        print_current_memory_usage("conv1")

        x = self.relu2(self.bn2(self.conv2(x)))
        print_current_memory_usage("conv2")
        x = F.max_pool2d(x, 2)

        x = self.relu3(self.bn3(self.conv3(x)))
        print_current_memory_usage("conv3")

        x = self.relu4(self.bn4(self.conv4(x)))
        print_current_memory_usage("conv4")
        x = F.max_pool2d(x, 2)

        x = self.relu5(self.bn5(self.conv5(x)))
        print_current_memory_usage("conv5")

        x = self.relu6(self.bn6(self.conv6(x)))
        print_current_memory_usage("conv6")
        x = F.max_pool2d(x, 2)

        x = self.relu7(self.bn7(self.conv7(x)))
        print_current_memory_usage("conv7")

        return x

    def forward_fc(self, x):
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        print_current_memory_usage("fc1")

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        print_current_memory_usage("fc2")

        x = self.fc3(x)
        print_current_memory_usage("fc3")

        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.forward_fc(x)

        return x


class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, CLASS_AMOUNT),
        )

    def forward(self, x):
        return self.vgg16(x)


# Initialize the model
model = CNNModel() if MODEL_USED == "CNN" else VGGModel()
image_X_Y_size_based_on_model = 224 if MODEL_USED == "VGG" else IMAGE_X_Y_SIZE

# Set the model to the desired dtype (all layers)
model.to(FC_DTYPE)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Display model summary
if SUMMARY_DISPLAYED:
    summary(
        model,
        input_size=(3, image_X_Y_size_based_on_model, image_X_Y_size_based_on_model),
    )

# Define transformations for the training and validation sets
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((image_X_Y_size_based_on_model, image_X_Y_size_based_on_model)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(10),
        v2.RandomPerspective(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# Load the dataset
train_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "training"), transform=transform
)
val_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "validation"), transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in tqdm(range(EPOCH_AMOUNT), desc="Epochs"):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        inputs = inputs.to(FC_DTYPE)
        with torch.autocast(device_type="cuda", dtype=CONV_DTYPE):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        running_loss += loss.item()

    end_time = time.time()
    iteration_time = end_time - start_time

    print(
        f"Epoch {epoch + 1}/{EPOCH_AMOUNT}, Loss: {running_loss / len(train_loader)}, Time: {iteration_time:.2f} seconds"
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
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
