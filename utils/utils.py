import copy
import torch
from torchvision import datasets, transforms
from sampling import cifar_iid, cifar_noniid, cifar_iid_unequal


def get_dataset(iid=1, unbalanced=0, num_users=100):
    """
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # mean and std of the CIFAR-10 dataset
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # mean and std of the CIFAR-10 dataset
    ])

    # choose the training and test datasets
    train_dataset = datasets.CIFAR10('data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('data', train=False,
                                    download=True, transform=transform_test)

    if iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(train_dataset, num_users)
    else:
        # Sample Non-IID user data from Mnist
        if unbalanced:
            # Chose unequal splits for every user
            user_groups = cifar_noniid_unbalanced(train_dataset, num_users)
        else:
            # Chose equal splits for every user
            user_groups = cifar_noniid(train_dataset, num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
