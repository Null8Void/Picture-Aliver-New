"""Setup script for Image2Video AI system."""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

setup(
    name="image2video-ai",
    version="1.0.0",
    description="Production Image to Video Synthesis System using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Engineering Team",
    author_email="ai@example.com",
    url="https://github.com/username/image2video",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "scipy>=1.11.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "diffusers>=0.25.0",
        "safetensors>=0.3.0",
        "timm>=0.9.0",
        "kornia>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "image2video=src.bin.cli:main",
            "picaliver=picture_aliver.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: OS Independent",
    ],
    keywords="ai computer-vision video-generation diffusion-models image-to-video",
    project_urls={
        "Bug Reports": "https://github.com/username/image2video/issues",
        "Source": "https://github.com/username/image2video",
        "Documentation": "https://github.com/username/image2video#readme",
    },
)