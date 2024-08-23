from dataset import AnimePortaraitsDataset
from model import SDXLModel

import yaml

with open("training_configs.yaml") as file:
    cfg = yaml.load(file, yaml.loader.SafeLoader)


model = SDXLModel(**cfg["model"])
train_dataset = AnimePortaraitsDataset(cfg['data']['parquet_file'])
val_dataset = AnimePortaraitsDataset(cfg['data']['parquet_file'], train=False)
model.finetune(train_dataset, val_dataset, cfg['data']['batch_size'], cfg['data']['drop_last'], cfg['epochs'])
