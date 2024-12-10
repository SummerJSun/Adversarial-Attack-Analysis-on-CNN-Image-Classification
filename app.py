import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from io import BytesIO
from models import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ifgsm_attack_batch(model, patches, label, epsilon, alpha, iters):
    patches = patches.clone().detach().to(device)
    label = label.to(device)
    patches.requires_grad = True

    for _ in range(iters):
        outputs = model(patches)
        loss = nn.CrossEntropyLoss()(outputs, label)
        model.zero_grad()
        loss.backward()
        grad = patches.grad.data.sign()
        patches = patches + alpha * grad
        patches = torch.clamp(patches, 0, 1)
        patches = torch.clamp(patches, patches - epsilon, patches + epsilon)
        patches = patches.detach()
        patches.requires_grad = True

    return patches

def process_image_with_slices(model, image, epsilon, alpha, iters):
    original_height, original_width = image.size
    transform_to_tensor = transforms.ToTensor()
    transform_to_image = transforms.ToPILImage()

    tensor_image = transform_to_tensor(image).unsqueeze(0)
    tensor_image = tensor_image.squeeze(0)

    padded_height = (tensor_image.shape[1] + 31) // 32 * 32
    padded_width = (tensor_image.shape[2] + 31) // 32 * 32
    padded_image = torch.zeros((3, padded_height, padded_width))
    padded_image[:, :tensor_image.shape[1], :tensor_image.shape[2]] = tensor_image
    perturbed_image = padded_image.clone()

    patches = []
    indices = []
    num_slices_y = padded_height // 32
    num_slices_x = padded_width // 32

    for i in range(num_slices_y):
        for j in range(num_slices_x):
            start_y, start_x = i * 32, j * 32
            patch = perturbed_image[:, start_y:start_y + 32, start_x:start_x + 32]
            patches.append(patch)
            indices.append((start_y, start_x))

    patches = torch.stack(patches).to(device)
    label = torch.zeros(len(patches), dtype=torch.long).to(device)

    perturbed_patches = ifgsm_attack_batch(model, patches, label, epsilon, alpha, iters)

    with torch.no_grad():
        for idx, (start_y, start_x) in enumerate(indices):
            perturbed_image[:, start_y:start_y + 32, start_x:start_x + 32] = perturbed_patches[idx]

    perturbed_image = perturbed_image[:, :tensor_image.shape[1], :tensor_image.shape[2]]
    perturbed_image = transform_to_image(perturbed_image)

    return perturbed_image


@st.cache_resource
def load_model(model_path):
    model = CNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model

st.title("Adversarial Image Generator with iFGSM on Slices")

model_path = "models/target_model.mod"
model = load_model(model_path)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    epsilon = st.slider("Select Epsilon (Strength of Perturbation)", 0.0, 1.0, 0.1, step=0.01)
    alpha = epsilon / 2
    iters = st.slider("Number of Iterations", 1, 50, 10)

    perturbed_image = process_image_with_slices(model, image, epsilon, alpha, iters)

    st.image(perturbed_image, caption="Perturbed Image with Original Resolution", use_column_width=True)

    buffer = BytesIO()
    perturbed_image.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button(
        label="Download Perturbed Image",
        data=buffer,
        file_name="perturbed_image.png",
        mime="image/png"
    )
