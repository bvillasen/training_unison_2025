# This script trains a deep convolutional neural network on the CIFAR-10 dataset. 
# The network has multiple convolutional layers, followed by fully connected layers, 
# which will put significant stress on the GPU in terms of both memory and compute.
# Increasing the batch size and the number of epochs will further increase the stress
# on the GPU.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

print('\nRunning simple pytorch example')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print( f'Pytorch is using device: {device} ')
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print( f'Number of GPUs: {n_gpus}')
batch_size = 256

# Define a large dataset and data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define a deep neural network
class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)  # Adjusted dimensions
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)  # Added maxpool to reduce dimensions
        x = x.view(-1, 256 * 4 * 4)  # Adjusted dimensions
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = DeepNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Increase the number of epochs for more stress
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

print("Training complete.")
