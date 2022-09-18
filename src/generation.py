from typing import Dict, Optional, Tuple

import PIL

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

class TextToImageGenerator:
    """
    Class for generating images from a text propmpt
    """

    MODEL_ID = "CompVis/stable-diffusion-v1-4"
    HEIGHT = 512
    WIDTH = 512
    MAX_TRIES = 5
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = self._get_model()
    
    def generate(
        self, 
        prompt: str, 
        num_inference_steps: int, 
        guidance_scale: float, 
        generator_seed: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Args:
            prompt (str)
                Text to generate the image from.
            num_inference_steps (int)
                Number of GAN iterations, suggested default for preview - 10, suggested default for generation - 50.
            guidance_scale (float)
                How stronly related the image needs to be to the prompt, higher values cause drop in image quality, suggested default - 7.5.
            generator_seed (torch.Tensor, *optional*)
                The seed of the random number generator used for generating the input latent state. Provides deterministic output. If None, the seed will be randomly sampled.
        """

        max_tries = self.MAX_TRIES if generator_seed is None else 1
        for try_id in range(self.MAX_TRIES):
            image, nsfw_detected, random_number_generator_seed = self._generate_image(prompt, num_inference_steps, guidance_scale, generator_seed)
            if nsfw_detected:
                continue
            return {
                'image': image,
                'seed': random_number_generator_seed,
            }
        return None
    
    def _get_model(self) -> StableDiffusionPipeline:
        model = StableDiffusionPipeline.from_pretrained(self.MODEL_ID, use_auth_token=True)
        return model.to(self.device)
    
    def _get_random_number_generator(self, seed: Optional[int]) -> Tuple[torch.Generator, int]:
        random_number_generator = torch.Generator(self.device)
        if seed is None:
            generator_seed = random_number_generator.seed()
        else:
            random_number_generator.manual_seed(seed)
            generator_seed = seed
        return random_number_generator, generator_seed
    
    def _generate_image(
        self, 
        prompt: str, 
        num_inference_steps: int, 
        guidance_scale: float, 
        generator_seed: Optional[int],
    ) -> Tuple[PIL.Image.Image, bool, int]:
        random_number_generator, random_number_generator_seed = self._get_random_number_generator(generator_seed)
        with autocast(self.device):
            model_output = self.model(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=random_number_generator,
                return_dict=True,
            )
        return model_output.images[0], model_output.nsfw_content_detected[0], random_number_generator_seed

class TextToImageGenerator:
    """
    Class for generating images from a text propmpt
    """
    MODEL_ID = "CompVis/stable-diffusion-v1-4"
    HEIGHT = 512
    WIDTH = 512
    MAX_TRIES = 5
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = self._get_model()
    
    def generate(
        self, 
        prompt: str, 
        num_inference_steps: int, 
        guidance_scale: float, 
        latents: Optional[torch.Tensor] = None
    ):
        """
        Args:
            prompt (str)
                Text to generate the image from.
            num_inference_steps (int)
                Number of GAN iterations, suggested default for preview - 10, suggested default for generation - 50.
            guidance_scale (float)
                How stronly related the image needs to be to the prompt, higher values cause drop in image quality, suggested default - 7.5.
            latents (torch.Tensor, *optional*)
                The initial latent state used by the model, provides deterministic output. If None, latents will be randomly sampled.
        """
        if latents is None:
            latents = self._get_latents()
        max_tries = self.MAX_TRIES if latents is None else 1
        for try_id in range(self.MAX_TRIES):
            image, nsfw_detected = self._generate_image(prompt, num_inference_steps, guidance_scale, latents)
            if nsfw_detected:
                continue
            return {
                'image': image,
                'latents': latents,
            }
        return None
    
    def _get_model(self):
        model = StableDiffusionPipeline.from_pretrained(self.MODEL_ID, use_auth_token=True)
        return model.to(self.device)
    
    def _get_latents(self):
        latents_shape = (1, self.model.unet.in_channels, self.HEIGHT // 8, self.WIDTH // 8)
        return torch.randn(latents_shape)
    
    def _generate_image(
        self, 
        prompt: str, 
        num_inference_steps: int, 
        guidance_scale: float, 
        latents: torch.Tensor
    ):
        with autocast(self.device):
            model_output = self.model(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                latents=latents,
                return_dict=True,
            )
        return model_output.images[0], model_output.nsfw_content_detected[0]