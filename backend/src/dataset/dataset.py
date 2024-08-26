import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from typing import Tuple


class AnimePortaraitsDataset(Dataset):
    """
    A custom Dataset for loading and processing anime portrait images and their captions.

    This dataset reads image paths and captions from a Parquet file and applies necessary
    transformations to the images.

    Attributes:
        imageSize (int): The size to which the images will be resized.
        transform (torchvision.transforms.Compose): The transformations to apply to the images.
        data : The DataFrame containing the image paths and captions.
    """

    def __init__(self, parquet_file: str, train: bool = True) -> None:
        """
        Initializes the AnimePortaraitsDataset.

        Args:
            parquet_file (str): The path to the Parquet file containing the dataset.
            train (bool, optional): Whether to use the training or evaluation split of the data.
                                    If True, use the first 20,000 samples for training.
                                    If False, use the remaining samples for evaluation.
                                    Defaults to True.

        """
        super().__init__()
        self.imageSize = 256
        self.transform = T.Compose(
            [
                T.Resize(self.imageSize),
                T.ToTensor(),
            ]
        )
        self.data = pd.read_parquet(parquet_file)
        if train:
            self.data = self.data[:20000]
        else:
            self.data = self.data[20000:]

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
        """
        Retrieves an image and its caption by index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, str]: A tuple containing:
                                      - The transformed image as a torch.Tensor.
                                      - The associated caption as a string.
        """
        record = self.data.iloc[index]
        image_path = record['id']
        caption = record['cap']

        # Open the image
        image = Image.open(image_path)
        image_tensor = self.transform(image)

        return image_tensor, caption
