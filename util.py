import torch
from models import CNN
from torchvision import transforms
from PIL import Image

def prediction_for_single_image(path_to_image: str):
    target_model = CNN(10)
    target_model.load_state_dict(torch.load('models/target_model.mod', weights_only=False))
    target_model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    input_image = Image.open(path_to_image)
    input_tensor = transform(input_image).unsqueeze(0)

    with torch.no_grad():
        output = target_model(input_tensor)

    _, predicted_class = torch.max(output, 1)
    print(f'Predicted class: {predicted_class.item()}')
    print(predicted_class.item().__class__)
    return predicted_class.item()
    
