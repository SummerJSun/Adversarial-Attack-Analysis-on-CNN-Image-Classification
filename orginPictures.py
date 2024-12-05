import os
from torchvision import datasets

def save_cifar10_test_images_by_class(output_folder):
    """
    Saves the CIFAR-10 test set images into subfolders based on their class labels.

    Parameters:
        output_folder (str): Path to the folder where test images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True)
    for idx, (image, label) in enumerate(test_set):
        class_folder = os.path.join(output_folder, str(label))
        os.makedirs(class_folder, exist_ok=True)
        image_path = os.path.join(class_folder, f"test_image_{idx:05d}.png")
        image.save(image_path)

    print(f"All CIFAR-10 test images have been saved in subfolders of {output_folder}")

save_cifar10_test_images_by_class(output_folder="./cifar10_test_images_by_class")
