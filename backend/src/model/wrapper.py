import torch
import torch.utils.data
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL, EulerDiscreteScheduler
import torch.utils.data.dataloader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from typing import List, Tuple, Callable
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console
import os
from PIL import Image


class SDXLModel:
    """
    A wrapper class for the Stable Diffusion XL (SDXL) model, which supports fine-tuning using LoRA (Low-Rank Adaptation).

    Attributes:
        pipeline (StableDiffusionXLImg2ImgPipeline): The Stable Diffusion XL image-to-image pipeline.
        unet (UNet2DConditionModel): The U-Net model used for image generation.
        vae (AutoencoderKL): The Variational Autoencoder used for encoding and decoding images.
        noise_scheduler (EulerDiscreteScheduler): The scheduler for adding noise to the latent space.
        lora_config (LoraConfig, optional): Configuration for LoRA adaptation.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
        console (Console): The console for displaying progress and messages.

    Args:
        model_name_or_path (str): The name of the pre-trained model from HF Hub or a path to a local model checkpoint. Default is 'Linaqruf/animagine-xl-3.0'.
        lora_weights (str, optional): Path to pre-trained LoRA weights. If not provided, a new LoRA configuration will be created.
        lora_rank (int, optional): The rank for LoRA. Default is 8.
        lora_alpha (int, optional): The alpha parameter for LoRA. Default is 32.
    """

    def __init__(
        self,
        model_name_or_path: str = 'Linaqruf/animagine-xl-3.0',
        lora_weights: str = None,
        lora_rank: int = 8,
        lora_alpha: int = 32,
    ):
        """
        The wrapper class for Stable Diffusion XL model, which can be finetuned using LoRA.

        Parameters:
            `model_name_or_path`: A model name available on HF Hub or a path to a local HF model checkpoint.
            `lora_weights`: the path to a pre-trained LoRA weights if available
            `lora_rank`: LoRA rank to apply
            `lora_alpha`: LoRA alpha to apply
        """
        self.pipeline: StableDiffusionXLImg2ImgPipeline = (
            StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_name_or_path, torch_dtype=torch.float16
            )
        )
        if torch.cuda.is_available():
            self.pipeline.to("cuda")

        # SD components
        self.unet: UNet2DConditionModel = self.pipeline.unet
        self.vae: AutoencoderKL = self.pipeline.vae
        self.vae.force_upcast = False

        self.noise_scheduler: EulerDiscreteScheduler = self.pipeline.scheduler

        # freeze all components and prepare for lora
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)

        if lora_weights:
            self.pipeline.load_lora_weights(lora_weights)
        else:
            self.lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )

            self.unet.add_adapter(self.lora_config)
        cast_training_params(self.unet)

        self.configure_optimization_scheme()
        self.console = Console()

    def encode_text(self, prompt: str | List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the prompts using the model text encoders."""
        prompt_embeds_list = []
        text_encoders = [self.pipeline.text_encoder, self.pipeline.text_encoder_2]
        tokenizers = [self.pipeline.tokenizer, self.pipeline.tokenizer_2]
        for i, text_encoder in enumerate(text_encoders):
            tokenizer = tokenizers[i]
            text_input_ids = tokenizer(
                prompt,
                padding='max_length',
                max_length=tokenizer.model_max_length,
                return_tensors='pt',
                truncation=True,
            ).input_ids

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    @staticmethod
    def compute_time_ids(original_size=(256, 256), crops_coords_top_left=(0, 0)):
        """
        Compute additional time IDs for the latent codes.

        Args:
            original_size (Tuple[int, int]): Size of the original image.
            crops_coords_top_left (Tuple[int, int]): Coordinates of the top-left crop.

        Returns:
            torch.Tensor: Tensor containing the computed time IDs.
        """
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (256, 256)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        return add_time_ids

    def encode_image_to_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images to latent codes using the VAE

        Parameters:
            `x`: batch of images, torch.Tensor of size [B, C, H, W]
        Returns:
            `out`: A tuple containing two objects of type torch.Tensor: the sampled latent code and the KL divergence
        """
        latent_dist = self.vae.encode(x).latent_dist
        return latent_dist.sample() * self.vae.config.scaling_factor, latent_dist.kl()

    def add_noise(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add random Gaussian noise to the latent codes

        Parameters:
            `z`: the latent code
        Returns:
            `out`: A tuple containing 3 objects of type torch.Tensor: the noised latent code, the timestep tensor and the noise
        """
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (z.shape[0],), device=z.device
        ).long()
        noise = torch.randn_like(z)
        return self.noise_scheduler.add_noise(z, noise, timesteps), timesteps, noise

    def calculate_loss(self, noise: torch.Tensor, predicted_noise: torch.Tensor):
        """
        Calculate the noise prediction loss of the model
        """
        return torch.nn.functional.mse_loss(predicted_noise, noise)

    def forward(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """
        Performs one forward pass

        Parameters:
            `images`: the images, scaled to (0, 1). A torch.Tensor with shape [B, C, H, W]
            `prompts`: prompts of the images. A list of strings.
        Returns:
            `loss`: The noise prediction loss of the Unet.
        """
        latent, kl = self.encode_image_to_latent(images)
        conditioning, add_text_embed = self.encode_text(prompts)

        add_time_ids = torch.cat([self.compute_time_ids()] * images.shape[0])

        noised_latent, timesteps, noise = self.add_noise(latent)
        predicted_noise = self.unet.forward(
            noised_latent,
            timesteps,
            encoder_hidden_states=conditioning.cuda(),
            return_dict=False,
            added_cond_kwargs={
                'text_embeds': add_text_embed.cuda(),
                'time_ids': add_time_ids.cuda(),
            },
        )[0]

        return self.calculate_loss(noise, predicted_noise)

    def configure_optimization_scheme(
        self,
        optimizer_class: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr: float = 1e-6,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.05,
        lr_scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
        **lr_scheduler_kwargs,
    ):
        """
        Configure the optimizer for the model (particularly the unet). The `optimizer` attribute will be created/updated.
        Parameters:
            `optimizer_class`: a class call for the `torch.optim.Optimizer`-inherited classes
            `lr`: the learning rate
            `betas`: beta values of the optimizer (for Adam optimizer family)
            `weight_decay`: the L2 regularization coefficient
            `lr_scheduler`: the LR scheduler to use with the optimizer. If `None`, a default LR scheduler will be created.
        Returns:
            None
        """
        if optimizer_class == torch.optim.SGD:
            self.optimizer = optimizer_class(
                self.unet.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer_class(
                self.unet.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
            )
        if lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=4000, eta_min=lr / 5
            )
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer, **lr_scheduler_kwargs)

    def configure_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        train: bool = True,
        batch_size: int = 4,
        drop_last: bool = True,
    ):
        """
        Create a dataloader

        Params:
            `dataset`: an object of any classes inherited from `torch.utils.data.Dataset`. The dataset must return two things when called: an image, and its prompt (caption).
            `train`: if this is a training dataloader
            `batch_size`: duh
            `drop_last`: if you want to drop the last batch for each epoch. Useful when used with `torch.compile` where usually static computation graphs are used.

        Returns:
            The dataloader with specified attributes.
        """
        return torch.utils.data.DataLoader(dataset, batch_size, shuffle=train, drop_last=drop_last)

    def save_checkpoint(self):
        print = self.console.print
        print("Saving checkpoint...")
        if not os.path.isdir("checkpoints"):
            os.mkdir("checkpoints")
        self.pipeline.save_lora_weights(
            "checkpoints", convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet))
        )
        print("Checkpoint saved")

    @torch.cuda.amp.autocast(dtype=torch.float16)
    def train_1epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int, total_epochs: int):
        """Helper method for training for 1 epoch"""
        print = self.console.print
        losses = []
        pbar = Progress(
            TextColumn(f"[green]Epoch {epoch}/{total_epochs}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("||"),
            TimeRemainingColumn(),
            TextColumn("loss = {task.fields[loss]:.4}"),
            console=self.console,
            transient=True,
        )

        task = pbar.add_task("", total=len(dataloader), loss=0.0)
        pbar.start()
        for img, prompts in dataloader:
            img = img.cuda()
            prompts = list(prompts)

            self.optimizer.zero_grad()

            loss = self.forward(img, prompts)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            losses.append(loss.item())
            pbar.update(task, advance=1, loss=loss.item())

        print(f"Epoch {epoch}, average loss = {sum(losses) / len(losses)}.")
        pbar.stop()

    @torch.cuda.amp.autocast(dtype=torch.float16)
    @torch.no_grad()
    def eval(self, dataloader: torch.utils.data.DataLoader, epoch: int, total_epochs: int):
        """Helper method for evaluating"""
        print = self.console.print
        losses = []
        pbar = Progress(
            TextColumn(f"[green]Evaluating, epoch {epoch}/{total_epochs}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("||"),
            TimeRemainingColumn(),
            TextColumn("loss = {task.fields[loss]:.4}"),
            console=self.console,
            transient=True,
        )

        task = pbar.add_task("", total=len(dataloader), loss=0.0)
        pbar.start()
        for img, prompts in dataloader:
            img = img.cuda()
            prompts = list(prompts)

            self.optimizer.zero_grad()

            loss = self.forward(img, prompts)

            losses.append(loss.item())
            pbar.update(task, advance=1, loss=loss.item())
        avg = sum(losses) / len(losses)
        print(f"Epoch {epoch}, average val loss = {avg}.")
        pbar.stop()
        return avg

    def finetune(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        batch_size: int = 4,
        drop_last: bool = False,
        epochs: int = 5,
    ):
        """
        Fine-tune the model with LoRA

        Parameters:
            `dataset`: an object of any classes inherited from `torch.utils.data.Dataset`. The dataset must return two things when called: an image, and its prompt (caption).
            `batch_size`: training batch size
            `drop_last`: if you want to drop the last batch for each epoch. Useful when used with `torch.compile` where usually static computation graphs are used.
            `epochs`: number of epochs to train
        """
        loss_min = 99.0
        train_dataloader = self.configure_dataloader(
            train_dataset, batch_size=batch_size, drop_last=drop_last
        )
        val_dataloader = self.configure_dataloader(
            val_dataset, batch_size=batch_size, drop_last=drop_last, train=False
        )
        for i in range(epochs):
            self.train_1epoch(train_dataloader, i, epochs)
            loss = self.eval(val_dataloader, i, epochs)
            if loss < loss_min:
                loss = loss_min
                self.save_checkpoint()

    def img2img(
        self, image: Image.Image, prompt: str, strength: float, **pipeline_kwargs
    ) -> Image.Image:
        """
        Image modification using SDXL

        All arguments should be compliant with `StableDiffusionXLImg2ImgPipeline.__call__`.

        Parameters:
            `image`: the original image to modify
            `prompt`: how to modify the image
            `strength`: how the modified result should differ from the original, float with value of range [0, 1]
            `pipeline_kwargs`: any other keyword argument accepted by `StableDiffusionXLImg2ImgPipeline.__call__`

        Returns:
            A PIL `Image`, the modifucation result.
        """
        x = self.pipeline.__call__(
            prompt, image=image, strength=strength, **pipeline_kwargs, return_dict=False
        )
        return x[0][0]
