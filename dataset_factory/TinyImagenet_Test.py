import os

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, ImageFolder

from .base import DatasetGenerator


class TINYIMAGENET_Generator(DatasetGenerator):
    def download(
        self,
    ) -> tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
        """
        Downloads the Tiny ImageNet dataset.

        Returns:
            tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
                A tuple containing the train and test Tiny ImageNet datasets.
        """
        # Get data
        if not os.path.exists(f"{self.rawdata_path}"):
            os.system(
                f"wget --directory-prefix {self.rawdata_path}/ http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            )
            os.system(
                f"unzip {self.rawdata_path}/tiny-imagenet-200.zip -d {self.rawdata_path}"
            )
        else:
            print("rawdata already exists.\n")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = ImageFolder_custom(
            root=os.path.join(self.rawdata_path, "tiny-imagenet-200", "train"),
            transform=transform,
        )
        testset = ImageFolder_custom(
            root=os.path.join(self.rawdata_path, "tiny-imagenet-200", "val"),
            transform=transform,
        )
        self.batch_size_train_cal = len(trainset)
        self.batch_size_test_cal = len(testset)
        return trainset, testset


class ImageFolder_custom(DatasetFolder):
    """
    A custom ImageFolder class for handling data indices.

    This class inherits from DatasetFolder and allows specifying data indices
    to select specific samples from the ImageFolder.
    """

    def __init__(
        self, root, dataidxs=None, train=True, transform=None, target_transform=None
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset.

        Args:
            index (int): The index of the sample.

        Returns:
            tuple: A tuple containing the sample and its target.
        """
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
