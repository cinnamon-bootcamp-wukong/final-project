import torch
import torch.utils.data
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL, EulerDiscreteScheduler
import torch.utils.data.dataloader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig

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
            model_name_or_path, torch_dtype=torch.float16, device='cuda'
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
        self.configure_optimization_scheme()
        self.console = Console()

    def encode_text(self, prompt: str | List[str]) -> torch.Tensor:
        tokens = self.text_tokenizer(prompt, return_tensors='pt', padding='True').to('cuda')
        return self.text_encoder(tokens.input_ids, return_dict=False)[0]

    def encode_image_to_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images to latent codes using the VAE

        Parameters:
            `x`: batch of images, torch.Tensor of size [B, C, H, W]
        Returns:
            `out`: A tuple containing two objects of type torch.Tensor: the sampled latent code and the KL divergence
        """
        latent_dist = self.vae.encode(x).latent_dist
        return latent_dist.sample(), latent_dist.kl()

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
        conditioning = self.encode_text(prompts)

        noised_latent, timesteps, noise = self.add_noise(latent)
        predicted_noise = self.unet.forward(
            noised_latent, timesteps, encoder_hidden_states=conditioning, return_dict=False
        )[0]

        return self.calculate_loss(noise, predicted_noise)

    def configure_optimization_scheme(
        self,
        optimizer_class: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr: float = 1e-5,
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
                self.optimizer, T_max=4000, eta_min=lr / 10
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

    @torch.cuda.amp.autocast(dtype=torch.float16)
    def train_1epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int, total_epochs: int):
        print = self.console.print
        losses = []
        pbar = Progress(
            TextColumn(f"[green]Epoch{epoch}/{total_epochs}"),
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

            self.optimizer.zero_grad()

            loss = self.forward(img, prompts)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            losses.append(loss.item())
            pbar.update(task, advance=1, loss=loss.item())

        print(f"Epoch {epoch}, average loss = {sum(losses) / len(losses)}.")
        print("Saving checkpoint...")
        if not os.path.isdir("checkpoints"):
            os.mkdir("checkpoints")
        self.unet.save_pretrained("checkpoints")
        print("Checkpoint saved")
        pbar.stop()

    def finetune(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 4,
        drop_last: bool = False,
        epochs: int = 5,
    ):
        dataloader = self.configure_dataloader(dataset, batch_size=batch_size, drop_last=drop_last)
        for i in range(epochs):
            self.train_1epoch(dataloader, i, epochs)

    def img2img(self, image: Image.Image, prompt: str, strength: float):
        self.pipeline.__call__(prompt, image, strength=strength)
