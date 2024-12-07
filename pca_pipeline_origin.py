import torch
import torchvision
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

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

# Determine axis limits for unified scaling
x_min, x_max = x_test_pca[:, 0].min(), x_test_pca[:, 0].max()
y_min, y_max = x_test_pca[:, 1].min(), x_test_pca[:, 1].max()

# Create output folder for class-specific PCA plots
output_folder = 'class_pca_plots_uniform'
os.makedirs(output_folder, exist_ok=True)

# Generate and save PCA plots for each class
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
    plt.title(f"PCA of CIFAR-10 Test Set: {class_name}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    # Set unified axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Save the plot
    save_path = os.path.join(output_folder, f"{class_name}_pca_uniform.png")
    plt.savefig(save_path)
    plt.close()

print(f"Class-specific PCA plots with unified axis scales saved in '{output_folder}' directory.")
