import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def main():
    batchsz = 32

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)
    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)


if __name__ == '__main__':
    print(torch.__version__)
    main()
