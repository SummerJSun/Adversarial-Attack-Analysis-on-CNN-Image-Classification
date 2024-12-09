import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import vit_b_16
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, test_loader, criterion, description="Evaluating"):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function.
        description (str): Progress bar description.

    Returns:
        avg_loss (float): Average loss on the dataset.
        accuracy (float): Accuracy on the dataset.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=description, leave=False)

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


def evaluate_on_multiple_epsilons(model, criterion, base_adv_folder, epsilons, transform, original_accuracy):
    """
    Evaluates the model on adversarial datasets for multiple epsilon values.

    Args:
        model (nn.Module): Trained model.
        criterion (nn.Module): Loss function.
        base_adv_folder (str): Base folder for adversarial datasets.
        epsilons (list): List of epsilon values.
        transform (Transform): Transform for adversarial datasets.
        original_accuracy (float): Accuracy on the original test set.

    Returns:
        epsilon_values (list): List of epsilon values.
        accuracy_drops (list): List of accuracy drops for each epsilon.
    """
    epsilon_values = []
    accuracy_drops = []

    for epsilon in epsilons:
        adv_folder = f"{base_adv_folder}_epsilon_{epsilon}"
        if os.path.exists(adv_folder):
            adv_test_set = torchvision.datasets.ImageFolder(root=adv_folder, transform=transform)
            adv_test_loader = DataLoader(adv_test_set, batch_size=128, shuffle=False)
            _, adv_test_accuracy = evaluate_model(
                model, adv_test_loader, criterion, description=f"Adversarial Test Set (epsilon={epsilon})"
            )
            accuracy_drop = original_accuracy - adv_test_accuracy
            epsilon_values.append(epsilon)
            accuracy_drops.append(accuracy_drop)
            print(f"Adversarial Test Set (epsilon={epsilon}) -> Accuracy Drop: {accuracy_drop:.2f}%")
        else:
            print(f"Adversarial folder for epsilon={epsilon} not found.")

    return epsilon_values, accuracy_drops


def plot_accuracy_drop(epsilon_values, accuracy_drops):
    """
    Plots accuracy drop against epsilon values.

    Args:
        epsilon_values (list): List of epsilon values.
        accuracy_drops (list): List of accuracy drops.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(epsilon_values, accuracy_drops, marker='o')
    plt.title("Accuracy Drop vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy Drop (%)")
    plt.grid(True)
    plt.savefig("vit_accuracy_drop.jpg")


def main():
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, 10)
    model.load_state_dict(torch.load("models/finetuned_vit.pth"))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, description="CIFAR-10 Test Set")
    print(f"CIFAR-10 Test Set -> Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    epsilon_values, accuracy_drops = evaluate_on_multiple_epsilons(
        model, criterion, "adversarial_cifar10", epsilons, transform_test, test_accuracy
    )

    plot_accuracy_drop(epsilon_values, accuracy_drops)


if __name__ == "__main__":
    main()
    
#   model.load_state_dict(torch.load("models/finetuned_vit.pth"))
# CIFAR-10 Test Set -> Loss: 1.2561, Accuracy: 53.98%
# Adversarial Test Set (epsilon=0.001) -> Accuracy Drop: 0.39%
# Adversarial Test Set (epsilon=0.005) -> Accuracy Drop: 0.36%
# Adversarial Test Set (epsilon=0.01) -> Accuracy Drop: 1.15%
# Adversarial Test Set (epsilon=0.05) -> Accuracy Drop: 5.12%
# Adversarial Test Set (epsilon=0.1) -> Accuracy Drop: 10.97%                                                                                                                                                         
# Adversarial Test Set (epsilon=0.5) -> Accuracy Drop: 36.18%
