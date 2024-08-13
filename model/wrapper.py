import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig

from typing import List, Tuple


class SDXLModel:
    def __init__(
        self,
        model_name_or_path: str = 'Linaqruf/animagine-xl-3.0',
        lora_rank: int = 8,
        lora_alpha: int = 32,
    ):
        """
        The wrapper class for Stable Diffusion XL model, which can be finetuned using LoRA.
        """
        self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16, device='cuda'
        )
        print(self.pipeline.unet)

        # SD components
        self.unet: UNet2DConditionModel = self.pipeline.unet
        self.vae: AutoencoderKL = self.pipeline.vae
        self.vae.force_upcast = False
        self.text_encoder: CLIPTextModel = self.pipeline.text_encoder
        self.text_tokenizer: CLIPTokenizer = self.pipeline.tokenizer

        self.noise_scheduler: EulerDiscreteScheduler = self.pipeline.scheduler

        # freeze all components and prepare for lora
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        self.unet.add_adapter(self.lora_config)

    def encode_text(self, prompt: str | List[str]) -> torch.Tensor:
        tokens = self.text_tokenizer(prompt, return_tensors='pt', padding='True')
        return self.text_encoder(tokens.input_ids, return_dict=False)[0]

    def encode_image_to_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent codes using the VAE
        Inputs:
        - `x`: batch of images, torch.Tensor of size [B, C, H, W]
        Returns
        - A tuple containing two objects of type torch.Tensor: the sampled latent code and the KL divergence
        """
        latent_dist = self.vae.encode(x).latent_dist
        return latent_dist.sample(), latent_dist.kl()
