import torch
import sys
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from models import CNN
import os

log_file = open('training_log.txt', 'w')
sys.stdout = log_file

D_train = torch.load('data/D_train.data', weights_only=False)
D_test = torch.load('data/D_test.data')

train_loader = DataLoader(D_train, batch_size=64, shuffle=True)
test_loader = DataLoader(D_test, batch_size=64, shuffle=False)  # Validation set for testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_model = CNN(num_classes=10).to(device)

def train_and_evaluate(model, train_loader, test_loader, epochs=256):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Testing
        model.eval()
        total_test_loss = 0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_test_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_accuracy = total_correct / total
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
        scheduler.step()

        # Early stopping condition
        if test_accuracy > 0.70 and avg_train_loss < 0.18:
            print("Stopping early as test accuracy has surpassed 70%.")
            break
    
    # Plotting the training and testing metrics
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Loss and Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig('CNN_performing_result.png')

def print_model_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data}")

output_folder = 'models'
os.makedirs(output_folder, exist_ok= True)

# Target model
train_and_evaluate(target_model, train_loader, test_loader)
print_model_weights(target_model)
torch.save(target_model.state_dict(), 'models/target_model.mod')

log_file.close()
