import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from models import CNN

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ifgsm_attack(model, images, labels, epsilon, alpha, iters):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    for _ in range(iters):
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.data.sign()
        images = images + alpha * grad
        images = torch.clamp(images, 0, 1)  # Keep pixel values in [0, 1]
        images = torch.clamp(images, images - epsilon, images + epsilon)  # Limit perturbation
        images = images.detach()
        images.requires_grad = True

    return images


def save_adversarial_images(dataset, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, (image, label) in enumerate(dataset):
        # Detach the image tensor and convert to NumPy
        image = image.detach().permute(1, 2, 0).numpy() * 255
        image = Image.fromarray(image.astype('uint8'))
        label_folder = os.path.join(output_folder, str(label))

        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        image.save(os.path.join(label_folder, f"{idx}.png"))


def create_adversarial_test_set(model_path, epsilon=0.03, alpha=0.01, iters=10, output_folder="adversarial_cifar10"):
    # Load CIFAR-10 test set
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Load trained model
    model = CNN(10)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval().to(device)

    adversarial_images = []
    adversarial_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Generate adversarial examples
        adv_images = ifgsm_attack(model, images, labels, epsilon, alpha, iters)
        adversarial_images.append(adv_images.cpu())
        adversarial_labels.extend(labels.cpu().numpy())

    # Combine all adversarial images
    adversarial_images = torch.cat(adversarial_images)

    # Save adversarial images
    save_adversarial_images(zip(adversarial_images, adversarial_labels), output_folder)
    print(f"Adversarial CIFAR-10 test set saved to {output_folder}")


def evaluate_model(model_path, test_loader, adv_test_loader, criterion):
    """
    Evaluates the CNN model on the original test set and adversarial test set.

    Args:
        model_path (str): Path to the trained model file.
        test_loader (DataLoader): DataLoader for the original test set.
        adv_test_loader (DataLoader): DataLoader for the adversarial test set.
        criterion (nn.Module): Loss function.

    Returns:
        None
    """
    # Load the trained model
    model = CNN(10)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval().to(device)

    def evaluate(loader):
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(loader)
        return avg_loss, accuracy

    # Evaluate on the original test set
    test_loss, test_accuracy = evaluate(test_loader)
    print(f"Original Test Set -> Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    # Evaluate on the adversarial test set
    adv_test_loss, adv_test_accuracy = evaluate(adv_test_loader)
    print(f"Adversarial Test Set -> Loss: {adv_test_loss:.4f}, Accuracy: {adv_test_accuracy:.2f}%")


if __name__ == "__main__":
    # Define parameters
    epsilon = 0.0001
    alpha = 0.00001
    iters = 10
    model_path = 'models/target_model.mod'
    output_folder = "adversarial_cifar10"

    # Create adversarial test set
    create_adversarial_test_set(
        model_path=model_path,
        epsilon=epsilon,
        alpha=alpha,
        iters=iters,
        output_folder=output_folder
    )

    # Load CIFAR-10 test set
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Load the adversarial CIFAR-10 dataset
    adv_test_set = torchvision.datasets.ImageFolder(root=output_folder, transform=transform)
    adv_test_loader = DataLoader(adv_test_set, batch_size=128, shuffle=False)

    # Define loss criterion
    criterion = nn.CrossEntropyLoss()

    # Evaluate model on both datasets
    evaluate_model(
        model_path=model_path,
        test_loader=test_loader,
        adv_test_loader=adv_test_loader,
        criterion=criterion
    )
