import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from typing import Tuple


class AnimePortaraitsDataset(Dataset):
    def __init__(self, parquet_file: str, train: bool = True) -> None:
        '''
        Dataset for portrait images crawled from Pixiv, with DreamBooth technique applied (trigger: `pixivvv`)
        '''
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
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
        record = self.data.iloc[index]
        image_path = record['id']
        caption = record['cap'] + ", pixivvv"

        # Open the image
        image = Image.open(image_path)
        image_tensor = self.transform(image)

        return image_tensor, caption
