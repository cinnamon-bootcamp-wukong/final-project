from src.model import SDXLModel
import gdown
import os


def test_dummy():
    print("Hello World!")


def test_gdown():
    id = "17xve1HRBAiDACOviOmGQbFNE8u2z4ZVi"
    try:
        os.mkdir("checkpoints/")
    except FileExistsError:
        pass

    gdown.download(id, "checkpoints/pytorch_lora_weights.safetensors")

    assert os.path.isfile("checkpoints/pytorch_lora_weights.safetensors")


def test_model():
    id = "17xve1HRBAiDACOviOmGQbFNE8u2z4ZVi"
    try:
        os.mkdir("checkpoints/")
    except FileExistsError:
        pass

    gdown.download(id, "checkpoints/pytorch_lora_weights.safetensors")

    SDXLModel(lora_weights="checkpoints")
