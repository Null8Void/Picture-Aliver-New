"""
Segmentation Module

Performs semantic and instance segmentation on input images.
Detects content types and provides masks for various object categories.
Supports human, furry, animal, and scene segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentType(str, Enum):
    """Types of content detected in images."""
    HUMAN = "human"
    FURRY = "furry"
    ANIMAL = "animal"
    LANDSCAPE = "landscape"
    OBJECT = "object"
    SCENE = "scene"
    UNKNOWN = "unknown"


@dataclass
class SegmentationResult:
    """
    Container for segmentation results.
    
    Attributes:
        mask: Segmentation mask
        categories: Detected categories
        content_type: Overall content type
        confidence: Confidence scores per category
        boxes: Bounding boxes for instances
        keypoints: Keypoints for characters
    """
    mask: torch.Tensor
    categories: List[str] = field(default_factory=list)
    content_type: ContentType = ContentType.UNKNOWN
    confidence: Optional[torch.Tensor] = None
    boxes: Optional[List[Tuple[int, int, int, int]]] = None
    keypoints: Optional[torch.Tensor] = None


@dataclass
class CategoryMask:
    """Mask for a specific category."""
    name: str
    mask: torch.Tensor
    confidence: float


class SegmentationModule:
    """
    Semantic segmentation with content type detection.
    
    Provides:
    - Multi-class semantic segmentation
    - Content type detection
    - Bounding box generation
    - Keypoint detection for characters
    
    Attributes:
        device: Compute device
        model_dir: Model directory
        num_classes: Number of segmentation classes
    """
    
    CATEGORY_COLORS = {
        "background": (0, 0, 0),
        "person": (255, 100, 100),
        "furry": (255, 150, 100),
        "animal": (255, 200, 100),
        "sky": (135, 206, 235),
        "grass": (34, 139, 34),
        "water": (30, 144, 255),
        "building": (128, 128, 128),
        "foliage": (34, 139, 34),
        "clothing": (200, 150, 100),
        "hair": (139, 90, 43),
    }
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_dir: Optional[Path] = None,
        num_classes: int = 19
    ):
        self.device = device or torch.device("cpu")
        self.model_dir = model_dir or Path("./models/segmentation")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        
        self.model: Optional[nn.Module] = None
        self.class_names: List[str] = []
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize segmentation model."""
        if self._initialized:
            return
        
        self.class_names = [
            "background", "person", "clothing", "hair", "skin",
            "furry", "tail", "ears", "wings",
            "sky", "grass", "foliage", "water", "ground",
            "building", "vehicle", "object", "animal", "other"
        ]
        
        self.model = SegmentationNetwork(
            num_classes=len(self.class_names),
            device=self.device
        )
        self.model.eval()
        
        self._initialized = True
    
    def segment(
        self,
        image: torch.Tensor,
        return_probs: bool = False
    ) -> SegmentationResult:
        """
        Perform segmentation on image.
        
        Args:
            image: Input tensor (C, H, W)
            return_probs: Whether to return probability maps
            
        Returns:
            SegmentationResult with masks and categories
        """
        if not self._initialized:
            self.initialize()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        original_size = image.shape[-2:]
        
        image_resized = F.interpolate(
            image,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )
        
        with torch.no_grad():
            output = self.model(image_resized)
            
            if return_probs:
                probs = F.softmax(output, dim=1)
                mask = probs.argmax(dim=1)
            else:
                mask = output.argmax(dim=1)
        
        mask = F.interpolate(
            mask.unsqueeze(1).float(),
            size=original_size,
            mode="nearest"
        ).squeeze(1)
        
        categories = self._extract_categories(mask)
        content_type = self._detect_content_type(categories)
        
        return SegmentationResult(
            mask=mask,
            categories=categories,
            content_type=content_type
        )
    
    def segment_with_prompts(
        self,
        image: torch.Tensor,
        prompts: List[str]
    ) -> SegmentationResult:
        """
        Segmentation with text prompts.
        
        Args:
            image: Input tensor
            prompts: List of category prompts
            
        Returns:
            SegmentationResult
        """
        result = self.segment(image, return_probs=True)
        
        for prompt in prompts:
            category = self._parse_prompt(prompt)
            if category:
                result.categories.append(category)
        
        return result
    
    def detect_content_type(
        self,
        image: torch.Tensor,
        segmentation: Optional[SegmentationResult] = None
    ) -> str:
        """
        Detect the overall content type of the image.
        
        Args:
            image: Input tensor
            segmentation: Pre-computed segmentation
            
        Returns:
            Content type string
        """
        if segmentation is None:
            segmentation = self.segment(image)
        
        return segmentation.content_type.value
    
    def _extract_categories(self, mask: torch.Tensor) -> List[str]:
        """Extract present categories from mask."""
        if mask.dim() > 2:
            mask = mask.squeeze()
        
        unique_classes = torch.unique(mask).tolist()
        
        categories = []
        for cls_idx in unique_classes:
            if cls_idx < len(self.class_names):
                categories.append(self.class_names[cls_idx])
        
        return categories
    
    def _detect_content_type(self, categories: List[str]) -> ContentType:
        """Determine content type from detected categories."""
        has_person = "person" in categories or "skin" in categories
        has_furry = "furry" in categories or "tail" in categories or "ears" in categories
        has_animal = "animal" in categories
        has_sky = "sky" in categories
        has_grass = "grass" in categories or "foliage" in categories
        has_water = "water" in categories
        has_building = "building" in categories
        
        if has_furry:
            return ContentType.FURRY
        elif has_person and not has_animal:
            return ContentType.HUMAN
        elif has_animal:
            return ContentType.ANIMAL
        elif has_sky and (has_grass or has_water or has_building):
            return ContentType.LANDSCAPE
        elif has_grass or has_water or has_building:
            return ContentType.SCENE
        elif categories and "object" in categories:
            return ContentType.OBJECT
        
        return ContentType.UNKNOWN
    
    def _parse_prompt(self, prompt: str) -> Optional[str]:
        """Parse category from prompt text."""
        prompt_lower = prompt.lower()
        
        mappings = {
            "person": ["person", "human", "character", "face", "body"],
            "furry": ["furry", "fur", "anthro"],
            "animal": ["animal", "creature", "beast"],
            "background": ["background", "bg", "scene"],
            "object": ["object", "item", "prop"],
        }
        
        for category, keywords in mappings.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    return category
        
        return None
    
    def create_visualization(
        self,
        mask: torch.Tensor,
        alpha: float = 0.7
    ) -> torch.Tensor:
        """
        Create colored visualization of segmentation.
        
        Args:
            mask: Segmentation mask (H, W)
            alpha: Blending factor for colors
            
        Returns:
            RGB visualization tensor
        """
        if mask.dim() > 2:
            mask = mask.squeeze()
        
        h, w = mask.shape
        
        visualization = torch.zeros(3, h, w, device=mask.device)
        
        for i, (name, color) in enumerate(self.CATEGORY_COLORS.items()):
            if i >= len(self.class_names):
                break
            
            class_idx = self.class_names.index(name) if name in self.class_names else -1
            if class_idx < 0:
                continue
            
            class_mask = (mask == class_idx)
            if class_mask.any():
                for c in range(3):
                    visualization[c][class_mask] = color[c] / 255.0
        
        return visualization
    
    def get_category_masks(
        self,
        mask: torch.Tensor
    ) -> List[CategoryMask]:
        """
        Get individual masks for each category.
        
        Args:
            mask: Combined segmentation mask
            
        Returns:
            List of CategoryMask objects
        """
        if mask.dim() > 2:
            mask = mask.squeeze()
        
        results = []
        
        for cls_idx in torch.unique(mask).tolist():
            if cls_idx >= len(self.class_names):
                continue
            
            class_name = self.class_names[cls_idx]
            
            class_mask = (mask == cls_idx).float()
            
            confidence = class_mask.mean().item()
            
            results.append(CategoryMask(
                name=class_name,
                mask=class_mask,
                confidence=confidence
            ))
        
        return results


class SegmentationNetwork(nn.Module):
    """
    U-Net based segmentation network.
    
    Lightweight segmentation model for content detection.
    Uses residual connections and attention for better feature extraction.
    
    Attributes:
        num_classes: Number of output classes
        device: Compute device
    """
    
    def __init__(self, num_classes: int, device: torch.device):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        
        self.encoder1 = self._make_encoder_block(3, 32)
        self.encoder2 = self._make_encoder_block(32, 64)
        self.encoder3 = self._make_encoder_block(64, 128)
        self.encoder4 = self._make_encoder_block(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = self._make_encoder_block(256, 512)
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder4 = self._make_encoder_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder3 = self._make_encoder_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = self._make_encoder_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder1 = self._make_encoder_block(64, 32)
        
        self.classifier = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Logits (B, num_classes, H, W)
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.up4(bottleneck)
        dec4 = self._crop_and_concat(dec4, enc4)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = self._crop_and_concat(dec3, enc3)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = self._crop_and_concat(dec2, enc2)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = self._crop_and_concat(dec1, enc1)
        dec1 = self.decoder1(dec1)
        
        output = self.classifier(dec1)
        
        return output
    
    def _make_encoder_block(
        self,
        in_channels: int,
        out_channels: int
    ) -> nn.Sequential:
        """Create an encoder block with residual connection."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def _crop_and_concat(
        self,
        upsampled: torch.Tensor,
        shortcut: torch.Tensor
    ) -> torch.Tensor:
        """Crop and concatenate skip connection."""
        _, _, h, w = upsampled.shape
        _, _, sh, sw = shortcut.shape
        
        dh = (sh - h) // 2
        dw = (sw - w) // 2
        
        cropped = shortcut[:, :, dh:dh+h, dw:dw+w]
        
        return torch.cat([upsampled, cropped], dim=1)