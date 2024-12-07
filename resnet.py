import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights  # Import ResNet-18 with pretrained weights

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_resnet_model(model, test_loader, criterion):
    """
    Evaluates the ResNet model on a given dataset.

    Args:
        model (nn.Module): The ResNet model.
        test_loader (DataLoader): DataLoader for the dataset to evaluate.
        criterion (nn.Module): Loss function.

    Returns:
        avg_loss (float): Average loss on the dataset.
        accuracy (float): Accuracy on the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_resnet_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Trains the ResNet model on a given dataset.

    Args:
        model (nn.Module): The ResNet model.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        num_epochs (int): Number of training epochs.

    Returns:
        model (nn.Module): Trained model.
    """
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model


def main():
    # Define the transform and load datasets
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ResNet requirements
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10 training and test sets
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Load perturbed (adversarial) dataset
    adv_test_set = torchvision.datasets.ImageFolder(root="adversarial_cifar10", transform=transform_test)
    adv_test_loader = DataLoader(adv_test_set, batch_size=128, shuffle=False)

    # Load pretrained ResNet-18 model from torchvision
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10 classes
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = train_resnet_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # Evaluate on original test set
    test_loss, test_accuracy = evaluate_resnet_model(model, test_loader, criterion)
    print(f"Original Test Set -> Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    # Evaluate on adversarial test set
    adv_test_loss, adv_test_accuracy = evaluate_resnet_model(model, adv_test_loader, criterion)
    print(f"Adversarial Test Set -> Loss: {adv_test_loss:.4f}, Accuracy: {adv_test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
