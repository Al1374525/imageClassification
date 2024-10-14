import torch
import torchvision
import torchvision.transforms as transforms

def get_loaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x : x.repeat(3, 1, 1)) ]) # Repeat 1 channel to 3 channels
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True)
    
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader