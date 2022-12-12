from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import PIL

import torch
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

class ImageGenerator(ABC):
    """
    Abstract class for image generators
    """

    MODEL_ID = "CompVis/stable-diffusion-v1-4"
    HEIGHT = 512
    WIDTH = 512
    MAX_TRIES = 5
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = self._get_model()
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        num_inference_steps: int, 
        guidance_scale: float, 
        generator_seed: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Optional[Dict]:
        pass
    
    def _get_random_number_generator(self, seed: Optional[int]) -> Tuple[torch.Generator, int]:
        random_number_generator = torch.Generator(self.device)
        if seed is None:
            generator_seed = random_number_generator.seed()
        else:
            random_number_generator.manual_seed(seed)
            generator_seed = seed
        return random_number_generator, generator_seed
    

class TextToImageGenerator(ImageGenerator):
    """
    Class for generating images from a text propmpt
    """
    
    def generate(
        self, 
        prompt: str, 
        num_inference_steps: int, 
        guidance_scale: float, 
        generator_seed: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Method for generating images.
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

class ImageToImageGenerator(ImageGenerator):
    """
    Class for generating images from a text propmpt and a reference image
    """
    
    def generate(
        self, 
        prompt: str, 
        reference_image: PIL.Image.Image,
        num_inference_steps: int, 
        guidance_scale: float, 
        noise_strength: float,
        generator_seed: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Method for generating images.
        Args:
            prompt (str)
                Text to generate the image from.
            reference_image (PIL.Image.Image)
                Starting point for the generated image, modified with random noise.
            num_inference_steps (int)
                Number of GAN iterations, suggested default for preview - 10, suggested default for generation - 50.
            guidance_scale (float)
                How stronly related the image needs to be to the prompt, higher values cause drop in image quality, suggested default - 7.5.
            noise_strength (float)
                How much noise shoule be added to the image, ergo - how strongly related the output is supposed to be to the reference image. Value from 0 - maximal similarity, to 1 - no relation. Suggested default - 0.75.
            generator_seed (torch.Tensor, *optional*)
                The seed of the random number generator used for generating the input latent state. Provides deterministic output. If None, the seed will be randomly sampled.
        """

        max_tries = self.MAX_TRIES if generator_seed is None else 1
        for try_id in range(self.MAX_TRIES):
            image, nsfw_detected, random_number_generator_seed = self._generate_image(
                prompt, 
                reference_image, 
                num_inference_steps, 
                guidance_scale, 
                noise_strength, 
                generator_seed
            )
            if nsfw_detected:
                continue
            return {
                'image': image,
                'seed': random_number_generator_seed,
            }
        return None
    
    def _get_model(self) -> StableDiffusionImg2ImgPipeline:
        model = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.MODEL_ID,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=True
        )
        return model.to(self.device)
    
    def _generate_image(
        self, 
        prompt: str, 
        reference_image: PIL.Image.Image,
        num_inference_steps: int, 
        guidance_scale: float, 
        noise_strength: float,
        generator_seed: Optional[int],
    ) -> Tuple[PIL.Image.Image, bool, int]:
        reference_image = reference_image.resize((self.WIDTH, self.HEIGHT))
        random_number_generator, random_number_generator_seed = self._get_random_number_generator(generator_seed)
        with autocast(self.device):
            model_output = self.model(
                prompt,
                init_image=reference_image, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=noise_strength, 
                generator=random_number_generator,
                return_dict=True,
            )
        return model_output.images[0], model_output.nsfw_content_detected[0], random_number_generator_seed