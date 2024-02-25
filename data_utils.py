import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np


class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return torchvision.transforms.functional.rotate(x, self.angle, fill=(0,))

def create_rotated_mnist_task(num_tasks, per_task_rotation, batch_size, transform=[]):
    train_loaders = []
    test_loaders = []

    g = torch.Generator()
    g.manual_seed(0)  # check: always setting generator to 0 ensures the same ordering of data

    for task in range(num_tasks):
        rotation_degree = (task-1) * per_task_rotation

        extended_transform = transform.copy()
        extended_transform.extend([
            RotationTransform(rotation_degree),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        transforms = torchvision.transforms.Compose(extended_transform)
        #print(transforms)

        train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
        test = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms)

        train_loader = torch.utils.data.DataLoader(train,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, generator=g)
        test_loader = torch.utils.data.DataLoader(test,  batch_size=256, shuffle=False, num_workers=0, pin_memory=True, generator=g)

        train_loaders.append({
            'loader':train_loader,
            'task':task,
            'rot':rotation_degree})
        test_loaders.append({
            'loader':test_loader,
            'task':task,
            'rot':rotation_degree})

    return train_loaders, test_loaders


class RotatedMNISTDataset(Dataset):
    '''
        This class provides MNIST images with random rotations sampled from
        a list of rotation angles. This list is dependent of the number of tasks
        `num_tasks` and the distance (measured in degrees) between tasks
        `per_task_rotation`.
    '''
    def __init__(self, root, train=True, transform=None, download=True, num_tasks=5, per_task_rotation=9):
        self.dataset = torchvision.datasets.MNIST(root=root, train=train, transform=transform, download=download)
        self.transform = transform
        self.rotation_angles = []
        for task in range(num_tasks):
            self.rotation_angles.append(float((task-1) * per_task_rotation))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        angle = np.random.choice(self.rotation_angles)  # Randomly choose a rotation angle
        rotated_image = F.rotate(image, angle, fill=(0,))
        return rotated_image, label, angle

def create_flattened_rotated_mnist_task(num_tasks, per_task_rotation, batch_size, transform=[]):
    g = torch.Generator()
    g.manual_seed(0)  # check: always setting generator to 0 ensures the same ordering of data

    extended_transform = transform.copy()
    extended_transform.extend([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    transforms = torchvision.transforms.Compose(extended_transform)
 

    train = RotatedMNISTDataset('./data', train=True, transform=transforms,
                                        num_tasks=num_tasks,
                                        per_task_rotation=per_task_rotation)
    test = RotatedMNISTDataset('./data', train=False, transform=transforms,
                                        num_tasks=num_tasks,
                                        per_task_rotation=per_task_rotation)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, generator=g)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, generator=g)

    return train_loader, test_loader