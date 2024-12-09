import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from PIL import Image
from models import CNN

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
        images = torch.clamp(images, 0, 1)
        images = torch.clamp(images, images - epsilon, images + epsilon)
        images = images.detach()
        images.requires_grad = True

    return images


def save_adversarial_images(dataset, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, (image, label) in enumerate(dataset):
        image = image.detach().permute(1, 2, 0).numpy() * 255
        image = Image.fromarray(image.astype('uint8'))
        label_folder = os.path.join(output_folder, str(label))

        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        image.save(os.path.join(label_folder, f"{idx}.png"))


def create_adversarial_test_set(model_path, epsilons, alphas, iters, base_output_folder="adversarial_cifar10"):

    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = CNN(10)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval().to(device)

    for epsilon, alpha in zip(epsilons, alphas):
        print(f"Creating adversarial examples with epsilon={epsilon}, alpha={alpha}")

        output_folder = f"{base_output_folder}_epsilon_{epsilon}"
        adversarial_images = []
        adversarial_labels = []

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            adv_images = ifgsm_attack(model, images, labels, epsilon, alpha, iters)
            adversarial_images.append(adv_images.cpu())
            adversarial_labels.extend(labels.cpu().numpy())

        adversarial_images = torch.cat(adversarial_images)

        save_adversarial_images(zip(adversarial_images, adversarial_labels), output_folder)
        print(f"Adversarial CIFAR-10 test set saved to {output_folder}")


if __name__ == "__main__":
    iters = 10
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    alphas = [e/2/iters for e in epsilons]
    
    model_path = 'models/target_model.mod'
    base_output_folder = "adversarial_cifar10"

    create_adversarial_test_set(
        model_path=model_path,
        epsilons=epsilons,
        alphas=alphas,
        iters=iters,
        base_output_folder=base_output_folder
    )
