import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor  # Import CLIP model and processor

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_clip_model(model, processor, test_loader, criterion):
    """
    Evaluates the CLIP image encoder on a given dataset.

    Args:
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor for preprocessing images.
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
            # Preprocess images using CLIP processor
            processed = processor(images, return_tensors="pt", padding=True).to(device)
            outputs = model.get_image_features(**processed)  # Get image features
            outputs = nn.Linear(outputs.size(-1), 10)(outputs)  # Linear layer for CIFAR-10 classes
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    # Define the transform and load datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Load original CIFAR-10 test set
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # Load perturbed (adversarial) dataset
    adv_test_set = torchvision.datasets.ImageFolder(root="adversarial_cifar10", transform=transform)
    adv_test_loader = DataLoader(adv_test_set, batch_size=128, shuffle=False)

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate on original test set
    test_loss, test_accuracy = evaluate_clip_model(model, processor, test_loader, criterion)
    print(f"Original Test Set -> Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    # Evaluate on adversarial test set
    adv_test_loss, adv_test_accuracy = evaluate_clip_model(model, processor, adv_test_loader, criterion)
    print(f"Adversarial Test Set -> Loss: {adv_test_loss:.4f}, Accuracy: {adv_test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
