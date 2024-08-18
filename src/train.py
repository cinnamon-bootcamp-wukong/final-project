from dataset import AnimePortaraitsDataset
from model import SDXLModel

import yaml

with open("training_configs.yaml") as file:
    cfg = yaml.load(file, yaml.loader.SafeLoader)


model = SDXLModel(**cfg["model"])
dataset = AnimePortaraitsDataset(cfg['data']['parquet_file'])
model.finetune(dataset, cfg['data']['batch_size'], cfg['data']['drop_last'], cfg['epochs'])
