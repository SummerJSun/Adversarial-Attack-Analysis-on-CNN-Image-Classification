import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm 

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
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_resnet_model(model, train_loader, criterion, optimizer, num_epochs=10, save_path="finetuned_resnet18.pth"):
    """
    Trains the ResNet model on a given dataset.

    Args:
        model (nn.Module): The ResNet model.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        num_epochs (int): Number of training epochs.
        save_path (str): File path to save the fine-tuned model.

    Returns:
        model (nn.Module): Trained model.
    """
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model


def main():
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    adv_test_set = torchvision.datasets.ImageFolder(root="adversarial_cifar10", transform=transform_test)
    adv_test_loader = DataLoader(adv_test_set, batch_size=128, shuffle=False)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_resnet_model(model, train_loader, criterion, optimizer, num_epochs=10, save_path="models/finetuned_resnet18.pth")

    test_loss, test_accuracy = evaluate_resnet_model(model, test_loader, criterion)
    print(f"Original Test Set -> Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    adv_test_loss, adv_test_accuracy = evaluate_resnet_model(model, adv_test_loader, criterion)
    print(f"Adversarial Test Set -> Loss: {adv_test_loss:.4f}, Accuracy: {adv_test_accuracy:.2f}%")


if __name__ == "__main__":
    main()


# Epoch [1/10], Loss: 182.7856, Accuracy: 84.06%
# Epoch [2/10], Loss: 109.7137, Accuracy: 90.46%
# Epoch [3/10], Loss: 82.7633, Accuracy: 92.71%
# Epoch [4/10], Loss: 65.6343, Accuracy: 94.25%
# Epoch [5/10], Loss: 54.2036, Accuracy: 95.20%
# Epoch [6/10], Loss: 45.3919, Accuracy: 95.99%
# Epoch [7/10], Loss: 39.1147, Accuracy: 96.59%
# Epoch [8/10], Loss: 34.8158, Accuracy: 96.87%
# Epoch [9/10], Loss: 29.4209, Accuracy: 97.41%
# Epoch [10/10], Loss: 26.4970, Accuracy: 97.71%
# Original Test Set -> Loss: 0.2598, Accuracy: 92.20%
# Adversarial Test Set -> Loss: 0.2961, Accuracy: 91.20%