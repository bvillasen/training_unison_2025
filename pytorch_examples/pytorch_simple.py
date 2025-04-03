import torch

print('\nRunning simple pytorch example')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print( f'Pytorch is using device: {device} ')

# Create random input and output data
x = torch.randn(100, 1).to(device)
y = 3 * x + 2 + torch.randn(100, 1).to(device)

# Define a simple linear model
model = torch.nn.Linear(1, 1).to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
  model.train()
  optimizer.zero_grad()
  
  # Forward pass
  y_pred = model(x)
  
  # Compute loss
  loss = criterion(y_pred, y)
  
  # Backward pass and optimization
  loss.backward()
  optimizer.step()
  
  if (epoch+1) % 100 == 0:
      print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# Print the learned parameters
for name, param in model.named_parameters():
  print(f'{name}: {param.data}')