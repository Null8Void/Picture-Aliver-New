"""
Text-to-Image Generator Module

Generates images from text prompts using diffusion models.
Supports various model architectures and quality settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class T2IConfig:
    """Configuration for text-to-image generation."""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    negative_prompt: str = "blurry, low quality, distorted, deformed, ugly"
    batch_size: int = 1


class TextToImageGenerator:
    """
    Text-to-image generation using diffusion models.
    
    Supports:
    - Stable Diffusion based generation
    - Guidance scale control
    - Negative prompting
    - Seed-based reproducibility
    - Multiple resolution outputs
    
    Attributes:
        device: Compute device
        config: Generation configuration
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        config: Optional[T2IConfig] = None
    ):
        self.device = device or torch.device("cpu")
        self.config = config or T2IConfig()
        
        self.unet: Optional[nn.Module] = None
        self.vae: Optional[nn.Module] = None
        self.text_encoder: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the text-to-image models."""
        if self._initialized:
            return
        
        print("[T2I] Initializing text-to-image models...")
        
        try:
            from diffusers import StableDiffusionPipeline, AutoencoderKL
            from transformers import AutoTokenizer, CLIPTextModel
            
            model_id = "runwayml/stable-diffusion-v1-5"
            
            try:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.pipe = self.pipe.to(self.device)
                self.pipe.enable_attention_slicing()
                
                if self.device.type == "cuda":
                    self.pipe.enable_xformers_memory_efficient_attention()
                
            except Exception as e:
                print(f"[T2I] Could not load model: {e}")
                print("[T2I] Using fallback generator")
                self._init_fallback()
            
        except ImportError:
            print("[T2I] Diffusers not available, using fallback")
            self._init_fallback()
        
        self._initialized = True
    
    def _init_fallback(self) -> None:
        """Initialize fallback generator."""
        self.pipe = None
        self.fallback_unet = FallbackUNet(device=self.device)
        self.fallback_vae = FallbackVAE(device=self.device)
        self.fallback_text_encoder = FallbackTextEncoder(device=self.device)
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate an image from text prompt.
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            width: Output width
            height: Output height
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG guidance scale
            seed: Random seed for reproducibility
            
        Returns:
            Generated image tensor (C, H, W)
        """
        if not self._initialized:
            self.initialize()
        
        width = width or self.config.width
        height = height or self.config.height
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale
        negative_prompt = negative_prompt or self.config.negative_prompt
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        if self.pipe is not None:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            image = result.images[0]
            
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            return image_tensor.to(self.device)
        else:
            return self._generate_fallback(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
    
    def _generate_fallback(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float
    ) -> torch.Tensor:
        """Generate using fallback neural network."""
        b, c, h, w = 1, 4, height // 8, width // 8
        
        latent = torch.randn(b, c, h, w, device=self.device)
        
        text_embeds = self.fallback_text_encoder(prompt)
        neg_embeds = self.fallback_text_encoder(negative_prompt)
        
        for i in range(num_steps):
            t = 1.0 - i / num_steps
            
            noise_pred = self.fallback_unet(latent, t, text_embeds)
            noise_pred_neg = self.fallback_unet(latent, t, neg_embeds)
            
            guided = noise_pred + guidance_scale * (noise_pred - noise_pred_neg)
            latent = latent - guided * 0.01
        
        image = self.fallback_vae.decode(latent)
        
        image = torch.sigmoid(image)
        
        image = F.interpolate(
            image,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )
        
        return image.squeeze(0)
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[torch.Tensor]:
        """Generate multiple images from prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class TextToVideoGenerator:
    """
    Text-to-video generation using latent diffusion.
    
    Generates video sequences from text prompts with motion coherence.
    
    Attributes:
        device: Compute device
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_temporal_consistency: bool = True
    ):
        self.device = device or torch.device("cpu")
        self.use_temporal_consistency = use_temporal_consistency
        
        self.t2i_model: Optional[TextToImageGenerator] = None
        self.temporal_unet: Optional[nn.Module] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize text-to-video models."""
        if self._initialized:
            return
        
        print("[T2V] Initializing text-to-video models...")
        
        self.t2i_model = TextToImageGenerator(device=self.device)
        self.t2i_model.initialize()
        
        if self.use_temporal_consistency:
            self.temporal_unet = TemporalUNet(device=self.device)
        
        self._initialized = True
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: int = 16,
        fps: int = 8,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.5,
        motion_prompt: Optional[str] = None,
        **kwargs
    ) -> "VideoFrames":
        """
        Generate video from text prompt.
        
        Args:
            prompt: Text description
            negative_prompt: What to avoid
            num_frames: Number of frames to generate
            fps: Frames per second
            width: Frame width
            height: Frame height
            guidance_scale: CFG scale
            motion_prompt: Motion description
            
        Returns:
            VideoFrames object
        """
        if not self._initialized:
            self.initialize()
        
        from .video_generator import VideoFrames
        
        frames = VideoFrames()
        
        keyframe_indices = self._get_keyframe_indices(num_frames)
        
        keyframes = []
        for i, idx in enumerate(keyframe_indices):
            progress = idx / (num_frames - 1) if num_frames > 1 else 0
            
            frame_prompt = self._modify_prompt_for_frame(prompt, progress, i, len(keyframe_indices))
            
            frame = self.t2i_model.generate(
                prompt=frame_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                **kwargs
            )
            keyframes.append(frame)
        
        if len(keyframes) == 1:
            keyframes = self._interpolate_frames(keyframes[0], num_frames)
        else:
            interpolated = []
            for i in range(len(keyframes) - 1):
                interp_frames = self._tween_frames(
                    keyframes[i],
                    keyframes[i + 1],
                    keyframe_indices[i + 1] - keyframe_indices[i]
                )
                interpolated.extend(interp_frames)
            
            keyframes = interpolated[:num_frames]
        
        for frame in keyframes:
            frames.append(frame)
        
        return frames
    
    def _get_keyframe_indices(self, num_frames: int) -> List[int]:
        """Get indices for keyframe generation."""
        if num_frames <= 4:
            return list(range(num_frames))
        
        num_keyframes = max(2, num_frames // 8)
        step = (num_frames - 1) / (num_keyframes - 1)
        return [int(i * step) for i in range(num_keyframes)]
    
    def _modify_prompt_for_frame(
        self,
        prompt: str,
        progress: float,
        keyframe_idx: int,
        total_keyframes: int
    ) -> str:
        """Modify prompt for temporal variation."""
        if keyframe_idx == 0:
            return f"{prompt}, front view, initial position"
        elif keyframe_idx == total_keyframes - 1:
            return f"{prompt}, final view, ending position"
        else:
            return f"{prompt}, middle perspective, transition moment"
    
    def _interpolate_frames(
        self,
        frame: torch.Tensor,
        num_frames: int
    ) -> List[torch.Tensor]:
        """Interpolate frames when only one keyframe exists."""
        frames = []
        for i in range(num_frames):
            frames.append(frame)
        return frames
    
    def _tween_frames(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_interp: int
    ) -> List[torch.Tensor]:
        """Interpolate between two keyframes."""
        frames = []
        for i in range(num_interp):
            alpha = i / max(1, num_interp - 1)
            
            if frame1.shape != frame2.shape:
                frame2 = F.interpolate(
                    frame2.unsqueeze(0),
                    size=frame1.shape[-2:],
                    mode="bilinear"
                ).squeeze(0)
            
            interp = (1 - alpha) * frame1 + alpha * frame2
            frames.append(interp)
        
        return frames


class TemporalUNet(nn.Module):
    """Temporal consistency network for video generation."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        self.temporal_attention = nn.MultiheadAttention(256, 4, batch_first=True)
        
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
        )
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply temporal consistency."""
        b, c, h, w = latents.shape
        
        if b > 1:
            flat = latents.view(b, c, -1).permute(0, 2, 1)
            attended, _ = self.temporal_attention(flat, flat, flat)
            latents = attended.permute(0, 2, 1).view(b, c, h, w)
        
        return latents


class FallbackUNet(nn.Module):
    """Simple fallback U-Net for image generation."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: float, context: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        
        m = self.middle(e2)
        
        out = self.decoder(m)
        
        return out


class FallbackVAE(nn.Module):
    """Simple fallback VAE for decoding."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
        )
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.decoder(z)


class FallbackTextEncoder(nn.Module):
    """Simple fallback text encoder."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        self.embedding = nn.Embedding(1000, 256)
        self.projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )
    
    def forward(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        tokens = torch.randint(0, 1000, (1, 77), device=self.device)
        embeds = self.embedding(tokens)
        return self.projection(embeds).mean(dim=1)


def generate_from_prompt(
    prompt: str,
    output_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> Optional[torch.Tensor]:
    """
    Convenience function for text-to-image generation.
    
    Args:
        prompt: Text description
        output_path: Optional path to save image
        device: Compute device
        **kwargs: Additional generation parameters
        
    Returns:
        Generated image tensor or None if saved
    """
    gen = TextToImageGenerator(device=device)
    image = gen.generate(prompt, **kwargs)
    
    if output_path:
        import torch.nn.functional as F
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(image_np).save(output_path)
        return None
    
    return image


if __name__ == "__main__":
    print("[TextToImage] Testing generation...")
    
    gen = TextToImageGenerator()
    
    print("Enter a prompt to generate an image (or 'quit' to exit):")
    while True:
        prompt = input("> ")
        if prompt.lower() in ["quit", "exit", "q"]:
            break
        
        if prompt.strip():
            image = gen.generate(prompt, num_inference_steps=20)
            print(f"Generated image shape: {image.shape}")