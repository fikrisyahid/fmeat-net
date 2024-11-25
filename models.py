import config
import helper
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CNNModel(nn.Module):
  def __init__(self):
    super(CNNModel, self).__init__()
    # First layer
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu1 = nn.ReLU()

    # Second layer
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.relu2 = nn.ReLU()

    # Third layer
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.relu3 = nn.ReLU()

    # Fourth layer
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
    self.bn4 = nn.BatchNorm2d(128)
    self.relu4 = nn.ReLU()

    # Fifth layer
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
    self.bn5 = nn.BatchNorm2d(256)
    self.relu5 = nn.ReLU()

    # Sixth layer
    self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
    self.bn6 = nn.BatchNorm2d(256)
    self.relu6 = nn.ReLU()

    # Seventh layer
    self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
    self.bn7 = nn.BatchNorm2d(512)
    self.relu7 = nn.ReLU()

    # Eight layer
    self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
    self.bn8 = nn.BatchNorm2d(512)
    self.relu8 = nn.ReLU()

    # Flatten layer
    self.flatten = nn.Flatten()

    # Fully connected layers
    self.fc1 = nn.Linear(512 * config.LAST_CONV_LAYER_IMAGE_SIZE, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, config.CLASS_AMOUNT)

    self.dropout = nn.Dropout(config.DROPOUT_RATE)

  def forward_conv(self, x):
    helper.print_current_memory_usage("conv1", before=True)
    x = self.relu1(self.bn1(self.conv1(x)))
    x = F.max_pool2d(x, 2)
    helper.print_current_memory_usage("conv")

    x = self.relu2(self.bn2(self.conv2(x)))
    x = self.relu3(self.bn3(self.conv3(x)))
    x = F.max_pool2d(x, 2)
    helper.print_current_memory_usage("conv2-3")

    x = self.relu4(self.bn4(self.conv4(x)))
    x = self.relu5(self.bn5(self.conv5(x)))
    x = F.max_pool2d(x, 2)
    helper.print_current_memory_usage("conv4-5")

    x = self.relu6(self.bn6(self.conv6(x)))
    x = self.relu7(self.bn7(self.conv7(x)))
    x = F.max_pool2d(x, 2)
    helper.print_current_memory_usage("conv6-7")

    x = self.relu8(self.bn8(self.conv8(x)))
    helper.print_current_memory_usage("conv8")

    return x

  def forward_fc(self, x):
    x = self.flatten(x)

    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    helper.print_current_memory_usage("fc1")

    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    helper.print_current_memory_usage("fc2")

    x = self.fc3(x)
    helper.print_current_memory_usage("fc3")

    return x

  def forward(self, x, epoch=0):
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
      nn.Linear(4096, config.CLASS_AMOUNT),
    )

  def forward(self, x, epoch=0):
    if epoch == 0:
      for param in self.vgg16.features.parameters():
        param.requires_grad = False
    if epoch == 5:
      for param in self.vgg16.features.parameters():
        param.requires_grad = True
    return self.vgg16(x)
