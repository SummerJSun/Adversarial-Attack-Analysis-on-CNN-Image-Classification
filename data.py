import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
D_non = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

D_tr, D_aux = random_split(dataset, [25000, 25000], generator=torch.Generator().manual_seed(42))
D_mem, _ = random_split(D_tr, [10000, 15000], generator=torch.Generator().manual_seed(42))

# save the dataset
torch.save(D_tr, 'training_data/D_tr.data')
torch.save(D_aux, 'training_data/D_aux.data')
torch.save(D_non, 'test_data/D_non.data')
torch.save(D_mem, 'test_data/D_mem.data')


