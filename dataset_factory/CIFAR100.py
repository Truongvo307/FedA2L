import torchvision
import torchvision.transforms as transforms

from .base import DatasetGenerator


class CIFAR100_Generator(DatasetGenerator):
    def download(
        self,
    ) -> tuple[torchvision.datasets.CIFAR100, torchvision.datasets.CIFAR100]:
        """
        Downloads the CIFAR100 dataset.

        Returns:
            tuple[torchvision.datasets.CIFAR100, torchvision.datasets.CIFAR100]:
                A tuple containing the train and test CIFAR100 datasets.
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = torchvision.datasets.CIFAR100(
            root=self.rawdata_path, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR100(
            root=self.rawdata_path, train=False, download=True, transform=transform
        )

        self.batch_size_train_cal = len(trainset.data)
        self.batch_size_test_cal = len(testset.data)
        return trainset, testset
