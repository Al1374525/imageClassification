import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_loaders


#Set the hyper parameters
batch_size= 32
num_epochs = 10
#@TODO - fine tune for achieving optimal learning rate to decrease loss
learning_rate = 0.001 


#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load data
train_loader, test_loader = get_loaders(batch_size)

# Define the model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # change the last layer to 10

#Loss and optimizer function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, 'Loss: {running_loss / len(train_loader)}")

#Evaulation
model.eval()
correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
accuracy = correct / len(test_loader.dataset)
print(f'Test Accuracy: {accuracy:.4f}')

