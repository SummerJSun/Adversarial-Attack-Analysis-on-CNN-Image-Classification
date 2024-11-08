import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image

def generate_white_image():
    """
    Generate one white image of size 32x32x3
    
    Returns:
    - PyTorch tensor of shape (3, 32, 32) with values in [0,1]
    """
    # Create one white image (255 for all channels)
    image = np.ones((32, 32, 3), dtype=np.uint8) * 255
    
    # Convert to PyTorch tensor and normalize to [0,1]
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    
    return tensor_image

def add_random_dot(image, seed):
    """
    Add a random black dot to an image
    
    Parameters:
    - image: PyTorch tensor of shape (3, 32, 32)
    - seed: random seed for reproducibility
    
    Returns:
    - PyTorch tensor of shape (3, 32, 32) with added dot
    """

    # Set random seed
    np.random.seed(seed)
    
    # Convert to numpy for easier manipulation
    img_np = image.numpy().transpose(1, 2, 0)  # CHW to HWC
    
    # Generate random position
    x = np.random.randint(0, 32)
    y = np.random.randint(0, 32)
    
    # Add black dot
    img_np[x, y, :] = 0
    
    # Convert back to PyTorch tensor
    return torch.from_numpy(img_np.transpose(2, 0, 1))  # HWC to CHW


# Example usage:
# Generate a white image
white_image = generate_white_image()
save_image(white_image, "white_image.png")

# # Add random dot to the image
# image_with_dot = add_random_dot(white_image)
# # Save both images
# save_image(image_with_dot, "white_image_with_dot.png")

current_image = white_image

num_dots = 10  # Change this to add more or fewer dots
for i in range(num_dots):
    current_image = add_random_dot(current_image, seed = i)  
    save_image(current_image, f"image_dots_{i+1}.png")
    
print(f"Generated {num_dots+1} images (including the original white image)")

print("Images saved in current directory")