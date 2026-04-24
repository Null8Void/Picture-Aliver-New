"""
Image Loader Module

Handles loading, preprocessing, and normalization of input images.
Supports various image formats and automatic resizing for optimal processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np
import torch
from PIL import Image


class ImageLoader:
    """
    Image loading and preprocessing utility.
    
    Handles various image formats, automatic resizing, normalization,
    and conversion to tensor format suitable for model inference.
    
    Attributes:
        target_size: Target size for longest edge (None = no resize)
        normalize: Whether to normalize to [-1, 1]
        device: Compute device
    """
    
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    
    def __init__(
        self,
        target_size: int = 512,
        normalize: bool = True,
        device: Optional[torch.device] = None
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.device = device or torch.device("cpu")
    
    def load(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load an image from path and convert to tensor.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tensor of shape (C, H, W) with values normalized
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {image_path.suffix}")
        
        image = Image.open(image_path)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        original_size = image.size
        
        if self.target_size and max(original_size) > self.target_size:
            scale = self.target_size / max(original_size)
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            image = image.resize(new_size, Image.LANCZOS)
        
        image_array = np.array(image)
        
        tensor = self._array_to_tensor(image_array)
        
        return tensor
    
    def load_from_array(self, image_array: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to tensor.
        
        Args:
            image_array: Array of shape (H, W, C) or (H, W)
            
        Returns:
            Tensor of shape (C, H, W)
        """
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]
        
        tensor = self._array_to_tensor(image_array)
        
        return tensor
    
    def _array_to_tensor(self, image_array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to normalized tensor."""
        if image_array.max() <= 1:
            image_array = (image_array * 255).astype(np.uint8)
        
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        
        if self.normalize:
            tensor = tensor / 127.5 - 1.0
        else:
            tensor = tensor / 255.0
        
        return tensor.to(self.device)
    
    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor back to PIL Image.
        
        Args:
            tensor: Tensor of shape (C, H, W)
            
        Returns:
            PIL Image
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if self.normalize:
            tensor = (tensor + 1.0) * 127.5
        else:
            tensor = tensor * 255.0
        
        tensor = tensor.clamp(0, 255).to(torch.uint8)
        
        array = tensor.permute(1, 2, 0).cpu().numpy()
        
        return Image.fromarray(array)
    
    def resize(
        self,
        tensor: torch.Tensor,
        size: Tuple[int, int],
        mode: str = "bilinear"
    ) -> torch.Tensor:
        """
        Resize tensor to target size.
        
        Args:
            tensor: Input tensor of shape (C, H, W) or (B, C, H, W)
            size: Target size (height, width)
            mode: Interpolation mode
            
        Returns:
            Resized tensor
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        resized = torch.nn.functional.interpolate(
            tensor,
            size=size,
            mode=mode,
            align_corners=False if mode == "bilinear" else None
        )
        
        if squeeze_output:
            resized = resized.squeeze(0)
        
        return resized
    
    def pad_to_multiple(
        self,
        tensor: torch.Tensor,
        multiple: int = 8
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Pad tensor to be divisible by multiple.
        
        Args:
            tensor: Input tensor
            multiple: Divisor for padding
            
        Returns:
            Tuple of (padded tensor, padding tuple)
        """
        _, h, w = tensor.shape
        
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        
        padded = torch.nn.functional.pad(tensor, padding, mode="reflect")
        
        return padded, padding
    
    def get_image_info(self, image_path: Union[str, Path]) -> dict:
        """
        Get information about an image without loading full data.
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with image metadata
        """
        image_path = Path(image_path)
        
        with Image.open(image_path) as img:
            return {
                "path": str(image_path),
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.size[0],
                "height": img.size[1],
                "aspect_ratio": img.size[0] / img.size[1]
            }