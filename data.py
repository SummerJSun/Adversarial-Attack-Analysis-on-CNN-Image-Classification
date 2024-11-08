import torch
from torchvision import datasets, transforms
import os 

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

output_folder = 'data'
os.makedirs(output_folder, exist_ok= True)

torch.save(train_dataset, 'data/D_tr.data')
torch.save(test_dataset, 'data/D_non.data')
