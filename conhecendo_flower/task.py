"""conhecendo-flower: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

# Normalize: normalizes each rgb channel using the formula (tensor - mean) / std
# ToTensor: converts a PIL Image or numpy.ndarray to tensor
# Componse: chains multiple transforms together
# Pytorch takes raw images, converts them to tensors, and normalizes them
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data. -> images divided into 10 classes 
    (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)"""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # IidPartitioner: Strategy to explain how to partition the data
        # The partitions are independent and identically distributed (IID)
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    # Each client loads its own partition
    # Its used in cross validation techniques
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    #shuffle=True: the data is reshuffled at every epoch
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        # I put running loss inside the epoch loop to reset it every epoch
        running_loss = 0.0
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            # Clear the gradients
            optimizer.zero_grad()
            # Forward pass: predict and compute loss
            # The loss are computed using the criterion function, we have the labels and the predictions
            # net(images) gives us predictions and labels are the true results.
            loss = criterion(net(images), labels)
            # Backward pass: copmute the gradients
            loss.backward()
            # Update the weights using the gradients using adam optimizer
            optimizer.step()
            # Accumulate the training loss
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            # Get the model prediction
            outputs = net(images)
            # Compute the loss
            loss += criterion(outputs, labels).item()
            # Verify how many predictions are correct
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy
