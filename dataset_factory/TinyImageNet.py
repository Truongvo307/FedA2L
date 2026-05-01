# /datasets/TinyImageNet.py

import os
import zipfile
import urllib.request
import shutil
from typing import Tuple

import torchvision
import torchvision.transforms as transforms
from PIL import Image

from .base import DatasetGenerator


class TINYIMAGENET_Generator(DatasetGenerator):
    """
    Handles the download and generation of the TinyImageNet dataset.

    TinyImageNet contains 200 classes. Each class has 500 training images,
    50 validation images, and 50 test images. We use the validation set
    as our test set since the official test set has no labels.
    """

    def _download_and_unzip(self):
        """Downloads and unzips the TinyImageNet dataset if not present."""
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(self.rawdata_path, "tiny-imagenet-200.zip")
        unzipped_path = os.path.join(self.rawdata_path, "tiny-imagenet-200")

        if os.path.exists(unzipped_path):
            print("TinyImageNet already downloaded and unzipped.")
            return

        os.makedirs(self.rawdata_path, exist_ok=True)
        print("Downloading TinyImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        print("Unzipping TinyImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.rawdata_path)
        os.remove(zip_path)
        print("Download and unzip complete.")

    def _reorganize_val_folder(self) -> str:
        """
        Reorganizes the validation folder to match the ImageFolder structure.

        The original validation folder has all images in one directory and a
        text file with annotations. This function creates subdirectories for
        each class and moves the images into them.

        Returns:
            str: Path to the reorganized validation folder.
        """
        val_dir = os.path.join(self.rawdata_path, "tiny-imagenet-200", "val")
        val_reorganized_dir = os.path.join(self.rawdata_path, "tiny-imagenet-200", "val_reorganized")

        if os.path.exists(val_reorganized_dir):
            return val_reorganized_dir

        print("Reorganizing validation folder for ImageFolder compatibility...")
        val_annotations_path = os.path.join(val_dir, "val_annotations.txt")
        val_images_dir = os.path.join(val_dir, "images")

        # Create mapping from class ID to images
        class_to_images = {}
        with open(val_annotations_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_name, class_id = parts[0], parts[1]
                if class_id not in class_to_images:
                    class_to_images[class_id] = []
                class_to_images[class_id].append(img_name)

        # Create reorganized directory structure
        os.makedirs(val_reorganized_dir, exist_ok=True)
        for class_id, img_names in class_to_images.items():
            class_dir = os.path.join(val_reorganized_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            for img_name in img_names:
                original_path = os.path.join(val_images_dir, img_name)
                new_path = os.path.join(class_dir, img_name)
                shutil.copyfile(original_path, new_path)

        print("Validation folder reorganization complete.")
        return val_reorganized_dir

    def download(
        self,
    ) -> Tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
        """
        Downloads and prepares the TinyImageNet dataset.

        Returns:
            A tuple containing the train and test (validation) datasets.
        """
        self._download_and_unzip()

        train_dir = os.path.join(self.rawdata_path, "tiny-imagenet-200", "train")
        val_dir_reorganized = self._reorganize_val_folder()

        # TinyImageNet images are 64x64. Normalization is standard for ImageNet.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Monkey-patching ImageFolder to handle potential corrupt images
        # torchvision.datasets.folder.IMG_EXTENSIONS += ('.jpeg',) # In case some images are .jpeg
        def pil_loader_robust(path):
            try:
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')
            except (IOError, OSError):
                print(f"Corrupt image: {path}, replacing with a blank image.")
                return Image.new('RGB', (64, 64))

        trainset = torchvision.datasets.ImageFolder(
            root=train_dir,
            transform=transform,
            loader=pil_loader_robust
        )
        testset = torchvision.datasets.ImageFolder(
            root=val_dir_reorganized,
            transform=transform,
            loader=pil_loader_robust
        )

        return trainset, testset