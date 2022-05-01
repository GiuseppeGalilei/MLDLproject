import copy
import torch
from torchvision import datasets, transforms
from sampling import cifar_iid, cifar_noniid, cifar_noniid_unbalanced


def get_datasets():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset

def get_user_groups(train_dataset, iid=True, unbalanced=False, tot_users=100):
    user_groups = None
    if iid:
        user_groups = cifar_iid(train_dataset, tot_users)
    else:
        if unbalanced:
            user_groups = cifar_noniid_unbalanced(train_dataset, tot_users)
        else:
            user_groups = cifar_noniid(train_dataset, tot_users)

    return user_groups

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg