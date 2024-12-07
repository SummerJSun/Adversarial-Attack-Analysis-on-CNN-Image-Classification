import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the saved CIFAR-10 dataset
train_dataset = torch.load('data/D_train.data')
test_dataset = torch.load('data/D_test.data')

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Convert data to flattened numpy arrays
x_test = torch.stack([test_dataset[i][0].view(-1) for i in range(len(test_dataset))]).numpy()
y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

# Normalize the test data
x_test_normalized = x_test / 255.0

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
x_test_pca = pca.fit_transform(x_test_normalized)

# Determine axis limits for unified scaling for original dataset
x_min, x_max = x_test_pca[:, 0].min(), x_test_pca[:, 0].max()
y_min, y_max = x_test_pca[:, 1].min(), x_test_pca[:, 1].max()

# Create output folder for PCA plots
output_folder = 'pca_plots'
os.makedirs(output_folder, exist_ok=True)

# Generate and save PCA plots for the original dataset
for class_id, class_name in enumerate(class_names):
    # Filter the data for the current class
    class_indices = (y_test == class_id)
    x_class_pca = x_test_pca[class_indices]
    y_class = y_test[class_indices]

    # Plot the PCA for the specific class
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        x_class_pca[:, 0],
        x_class_pca[:, 1],
        c=[class_id] * len(x_class_pca),
        cmap='tab10',
        alpha=0.7,
        vmin=0,
        vmax=9,
    )
    plt.colorbar(scatter, ticks=range(10), label='Classes')
    plt.title(f"PCA of CIFAR-10 Test Set: {class_name} (Original)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    # Set unified axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Save the plot
    save_path = os.path.join(output_folder, f"{class_name}_pca_original.png")
    plt.savefig(save_path)
    plt.close()

print(f"Original dataset PCA plots saved in '{output_folder}' directory.")

########################################################################################
# adversarially attacked
attacked_image_path = './adversarial_cifar10' 
# Load adversarially attacked images
x_attacked = []
y_attacked = []

for class_id, class_name in enumerate(range(10)):
    class_folder = os.path.join(attacked_image_path, str(class_id))
    if not os.path.exists(class_folder):
        print(f"Warning: Folder '{class_folder}' does not exist.")
        continue

    for image_file in os.listdir(class_folder):
        if image_file.endswith('.png'):
            # Load the image and convert to a flattened array
            image_path = os.path.join(class_folder, image_file)
            image = Image.open(image_path).convert('RGB')  # Ensure 3-channel RGB
            image_array = np.array(image).flatten() / 255.0  # Normalize pixel values
            x_attacked.append(image_array)
            y_attacked.append(class_id)

# Convert lists to numpy arrays
x_attacked = np.array(x_attacked)
y_attacked = np.array(y_attacked)


# do a new PCA to the attacked dataset
pca_attacked = PCA(n_components=2)
x_attacked_pca = pca_attacked.fit_transform(x_attacked)

#set the axis limits for the attacked dataset
x_min, x_max = x_attacked_pca[:, 0].min(), x_attacked_pca[:, 0].max()
y_min, y_max = x_attacked_pca[:, 1].min(), x_attacked_pca[:, 1].max()

# Generate and save PCA plots for adversarially attacked dataset
for class_id, class_name in enumerate(class_names):
    # Filter the data for the current class
    class_indices = (y_attacked == class_id)
    x_class_pca = x_attacked_pca[class_indices]
    y_class = y_attacked[class_indices]

    # Plot the PCA for the specific class
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        x_class_pca[:, 0],
        x_class_pca[:, 1],
        c=[class_id] * len(x_class_pca),
        cmap='tab10',
        alpha=0.7,
        vmin=0,
        vmax=9,
    )

    # Set unified axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.colorbar(scatter, ticks=range(10), label='Classes')
    plt.title(f"PCA of CIFAR-10 Test Set: {class_name} (Adversarially Attacked)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Save the plot
    save_path = os.path.join(output_folder, f"{class_name}_pca_attacked.png")
    plt.savefig(save_path)
    plt.close()

print(f"Adversarially attacked dataset PCA plots saved in '{output_folder}' directory.")

########################################################################################
#do a pca to the attcked and original dataset together
#start from load data
test_dataset2 = torch.load('data/D_test.data')
x_test2 = torch.stack([test_dataset2[i][0].view(-1) for i in range(len(test_dataset2))]).numpy()
y_test2 = np.array([test_dataset2[i][1] for i in range(len(test_dataset2))])

x_test2 = x_test2 / 255.0

attacked_image_path2 = './adversarial_cifar10' 
# Load adversarially attacked images
x_attacked2 = []
y_attacked2 = []
for class_id, class_name in enumerate(range(10)):
    class_folder = os.path.join(attacked_image_path2, str(class_id))
    if not os.path.exists(class_folder):
        print(f"Warning: Folder '{class_folder}' does not exist.")
        continue

    for image_file in os.listdir(class_folder):
        if image_file.endswith('.png'):
            # Load the image and convert to a flattened array
            image_path = os.path.join(class_folder, image_file)
            image = Image.open(image_path).convert('RGB')  # Ensure 3-channel RGB
            image_array = np.array(image).flatten() / 255.0  # Normalize pixel values
            x_attacked2.append(image_array)
            y_attacked2.append(class_id)

# Convert lists to numpy arrays
x_attacked2 = np.array(x_attacked2)
y_attacked2 = np.array(y_attacked2)

#concatenate the original and attacked dataset
x_combined = np.concatenate((x_test2, x_attacked2))
y_combined = np.concatenate((y_test2, y_attacked2))

pca_combined = PCA(n_components=2)
x_combined_pca = pca_combined.fit_transform(x_combined)

#set the axis limits for the combined dataset
x_min, x_max = x_combined_pca[:, 0].min(), x_combined_pca[:, 0].max()
y_min, y_max = x_combined_pca[:, 1].min(), x_combined_pca[:, 1].max()

# Plot PCA for combined dataset with color-coded points (original in blue, attacked in red)
for class_id, class_name in enumerate(class_names):
    # Filter data for the current class
    original_indices = (y_test == class_id)
    attacked_indices = (y_attacked == class_id)

    x_original_class = x_test_pca[original_indices]
    x_attacked_class = x_attacked_pca[attacked_indices]

    # Plot the PCA for the specific class
    plt.figure()

    # Plot attacked points in red
    plt.scatter(
        x_attacked_class[:, 0],
        x_attacked_class[:, 1],
        color='red',
        alpha=0.7,
        label='Attacked',
    )
    
    # Plot original points in blue
    plt.scatter(
        x_original_class[:, 0],
        x_original_class[:, 1],
        color='blue',
        alpha=0.7,
        label='Original',
    )
    

    
    # Add legend to differentiate
    plt.legend()
    
    # Set plot title and axis labels
    plt.title(f"PCA of CIFAR-10 Test Set: {class_name} (Combined)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Save the plot
    save_path = os.path.join(output_folder, f"{class_name}_pca_combined_colored.png")
    plt.savefig(save_path)
    plt.close()

print(f"Color-coded combined dataset PCA plots saved in '{output_folder}' directory.")



