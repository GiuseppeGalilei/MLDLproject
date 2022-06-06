from torchvision import transforms, datasets
from torch.utils.data import Dataset

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]


def get_datasets(augmentation=False):
    trainset = datasets.CIFAR10("./data", train=True, download=True)
    testset = datasets.CIFAR10("./data", train=False, download=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std),])
    if augmentation:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)])
        trainset.transform = transform_train
    else:
        trainset.transfrom = transform
    testset.transform = transform

    return trainset, testset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label