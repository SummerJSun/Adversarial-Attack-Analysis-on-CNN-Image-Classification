import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import os
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_clip_zero_shot(model, preprocess, class_names, data_loader, description="Evaluating"):
    """
    Evaluates the zero-shot capabilities of CLIP's image encoder on a dataset.

    Args:
        model: CLIP model.
        preprocess: Preprocessing function for input images.
        class_names: List of class names as strings.
        data_loader: DataLoader for the test dataset.
        description: Progress bar description.

    Returns:
        accuracy: Accuracy on the dataset.
    """
    model.eval()
    correct = 0
    total = 0

    class_texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=description, leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(class_texts)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            predicted = logits.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    return accuracy


def evaluate_on_adversarial_epsilons(model, preprocess, class_names, base_adv_folder, epsilons, transform, original_accuracy):
    """
    Evaluates CLIP's zero-shot capabilities on adversarial datasets for multiple epsilon values.

    Args:
        model: CLIP model.
        preprocess: Preprocessing function for input images.
        class_names: List of class names as strings.
        base_adv_folder: Base folder for adversarial datasets.
        epsilons: List of epsilon values.
        transform: Transform for adversarial datasets.
        original_accuracy: Accuracy on the original test set.

    Returns:
        epsilon_values: List of epsilon values.
        accuracy_drops: List of accuracy drops for each epsilon.
    """
    epsilon_values = []
    accuracy_drops = []

    for epsilon in epsilons:
        adv_folder = f"{base_adv_folder}_epsilon_{epsilon}"
        if os.path.exists(adv_folder):
            adv_test_set = torchvision.datasets.ImageFolder(root=adv_folder, transform=transform)
            adv_test_loader = DataLoader(adv_test_set, batch_size=128, shuffle=False)
            adv_test_accuracy = evaluate_clip_zero_shot(
                model, preprocess, class_names, adv_test_loader, description=f"Adversarial Test Set (epsilon={epsilon})"
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
        epsilon_values: List of epsilon values.
        accuracy_drops: List of accuracy drops.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(epsilon_values, accuracy_drops, marker='o')
    plt.title("Accuracy Drop vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy Drop (%)")
    plt.grid(True)
    plt.savefig("clip_accuracy_drop.jpg")


def main():
    model, preprocess = clip.load("ViT-B/32", device=device)

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    test_accuracy = evaluate_clip_zero_shot(model, preprocess, class_names, test_loader, description="CIFAR-10 Test Set")
    print(f"CIFAR-10 Test Set -> Zero-Shot Accuracy: {test_accuracy:.2f}%")

    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    epsilon_values, accuracy_drops = evaluate_on_adversarial_epsilons(
        model, preprocess, class_names, "adversarial_cifar10", epsilons, transform_test, test_accuracy
    )

    plot_accuracy_drop(epsilon_values, accuracy_drops)


if __name__ == "__main__":
    main()

# Files already downloaded and verified
# CIFAR-10 Test Set -> Zero-Shot Accuracy: 84.71%
# Adversarial Test Set (epsilon=0.001) -> Accuracy Drop: 0.64%                                                                                                                                                        
# Adversarial Test Set (epsilon=0.005) -> Accuracy Drop: 0.61%
# Adversarial Test Set (epsilon=0.01) -> Accuracy Drop: 2.13%
# Adversarial Test Set (epsilon=0.05) -> Accuracy Drop: 16.30%
# Adversarial Test Set (epsilon=0.1) -> Accuracy Drop: 37.07%                                                                                                                                                         
# Adversarial Test Set (epsilon=0.5) -> Accuracy Drop: 78.21%