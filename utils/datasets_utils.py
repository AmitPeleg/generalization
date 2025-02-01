import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

from settings import DATA_DIR


def get_dataset(config, num_samples, classes_seed, permutation_seed, aug='none', device='cuda'):
    name = config['dataset.name']
    if name in ["mnist", "cifar10"]:
        dataset = Dataset_Mnist_Cifar10(batch_size=num_samples, aug=aug, train_count=num_samples,
                                        num_classes=config['dataset.mnistcifar.num_classes'], classes_seed=classes_seed,
                                        permutation_seed=permutation_seed, dataset_name=name)
        train_data, train_labels, test_data, test_labels, test_all_data, test_all_labels = \
            process_dataset_mnist_cifar(dataset, device)
    else:
        raise ValueError(f"dataset name {name} is not supported")

    classes = torch.unique(test_all_labels)
    return train_data, train_labels, test_data, test_labels, test_all_data, test_all_labels, classes


def process_dataset_mnist_cifar(dataset, device):
    train_data, train_labels = next(iter(dataset.train))
    test_data, test_labels = next(iter(dataset.test))
    train_data, train_labels, test_data, test_labels = train_data.to(device), train_labels.to(device), test_data.to(
        device), test_labels.to(device)
    test_all_data, test_all_labels = dataset.test_all_data, dataset.test_all_labels
    return train_data, train_labels, test_data, test_labels, test_all_data, test_all_labels


class Dataset_Mnist_Cifar10:
    def __init__(self, batch_size, threads=1, aug='none', train_count=None, num_classes=2, classes_seed=10,
                 permutation_seed=201, dataset_name='mnist'):
        self.dataset_name = dataset_name
        mean, std = self._get_statistics()
        # print(f"{mean=} {std=}")
        torch.manual_seed(classes_seed)
        if aug == "none":
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif aug == "for_presentation":
            train_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            raise NotImplementedError

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        if dataset_name == 'mnist':
            complete_train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True,
                                                            transform=train_transform)
            complete_test_set = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True,
                                                           transform=test_transform)
        elif dataset_name == 'cifar10':
            complete_train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True,
                                                              transform=train_transform)
            complete_test_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True,
                                                             transform=test_transform)
        else:
            raise NotImplementedError

        if dataset_name == 'mnist':
            labels = complete_train_set.targets.clone().detach()
            labels_test = complete_test_set.targets.clone().detach()
        elif dataset_name == 'cifar10':
            labels = torch.tensor(complete_train_set.targets)
            labels_test = torch.tensor(complete_test_set.targets)
        else:
            raise NotImplementedError

        new_labels = -torch.ones_like(labels)
        new_labels_test = -torch.ones_like(labels_test)

        train_indices_list = []
        val_indices_list = []
        test_indices_list = []

        num_original_classes = len(complete_test_set.classes)
        classes_list = torch.arange(num_original_classes)[torch.randperm(num_original_classes)][:num_classes]
        if permutation_seed != classes_seed:
            torch.manual_seed(permutation_seed)
        for i, cur_class in enumerate(classes_list):
            indices_of_cur_class = torch.arange(len(labels))[labels == cur_class]
            new_labels[labels == cur_class] = i
            indices_len = len(indices_of_cur_class)
            indices_of_cur_class = indices_of_cur_class[torch.randperm(indices_len)]
            val_indices_list.append(indices_of_cur_class[:256 // num_classes])
            train_indices_list.append(
                indices_of_cur_class[256 // num_classes:256 // num_classes + train_count // num_classes])

            indices_of_cur_class_test = torch.arange(len(labels_test))[labels_test == cur_class]
            new_labels_test[labels_test == cur_class] = i
            indices_len_test = len(indices_of_cur_class_test)
            indices_of_cur_class_test = indices_of_cur_class_test[torch.randperm(indices_len_test)]
            test_indices_list.append(indices_of_cur_class_test)

        complete_train_set.targets = new_labels
        complete_test_set.targets = new_labels_test
        val_indices = torch.cat(val_indices_list, dim=0)
        train_indices = torch.cat(train_indices_list, dim=0)
        test_indices = torch.cat(test_indices_list, dim=0)
        train_set = torch.utils.data.Subset(
            complete_train_set,
            train_indices
        )
        val_set = torch.utils.data.Subset(
            complete_train_set,
            val_indices
        )
        test_set = torch.utils.data.Subset(
            complete_test_set,
            test_indices
        )
        if aug == "for_presentation":
            self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                                     num_workers=threads)
        else:
            self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                     num_workers=threads)
        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.val = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test_all_data, self.test_all_labels = zip(*[(x[None, :], y) for x, y in test_set])
        self.test_all_data = torch.cat(self.test_all_data, dim=0)
        self.test_all_labels = torch.tensor(self.test_all_labels)

        if dataset_name == 'cifar10':
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        if self.dataset_name == 'mnist':
            train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True,
                                                   transform=transforms.ToTensor())
        elif self.dataset_name == 'cifar10':
            train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True,
                                                     transform=transforms.ToTensor())
        else:
            raise NotImplementedError
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
