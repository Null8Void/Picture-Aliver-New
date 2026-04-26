# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller build specification for Picture-Aliver Desktop (PyQt5)

Build commands:
    cd /path/to/picture-aliver
    python -m PyInstaller desktop/pyqt/build.spec --noconfirm

Output:
    dist/Picture-Aliver/Picture-Aliver.exe
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Use forward slashes to avoid escape sequence issues
PROJECT_ROOT = Path("D:/Git/Picture-Aliver")
SCRIPT_PATH = PROJECT_ROOT / "desktop" / "pyqt" / "main.py"

# Collect all src modules
src_modules = []
src_dir = PROJECT_ROOT / "src"
if src_dir.exists():
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" not in str(py_file):
            src_modules.append((str(py_file), "src"))

# Import PyInstaller utilities for collecting binaries
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files
import os

# For PyTorch on Windows, we need to include the DLLs from torch\lib folder
torch_lib_path = os.path.join(os.sys.prefix, "torch", "lib")
torch_dlls = []
if os.path.exists(torch_lib_path):
    for f in os.listdir(torch_lib_path):
        if f.endswith(".dll"):
            torch_dlls.append((os.path.join(torch_lib_path, f), "torch/lib"))

# Collect all torch and torchvision data including DLLs
torch_all = collect_all('torch')
torchvision_all = collect_all('torchvision')

# collect_all returns (datas, binaries, zipfiles, remote_ui) - need both
torch_datas = torch_all[0] if torch_all else []
torch_binaries = torch_all[1] if len(torch_all) > 1 else []
torchvision_datas = torchvision_all[0] if torchvision_all else []
torchvision_binaries = torchvision_all[1] if len(torchvision_all) > 1 else []

# Add torch DLLs explicitly
all_binaries = list(torch_binaries) + list(torchvision_binaries) + torch_dlls

# Collect numpy and other critical binaries
numpy_all = collect_all('numpy')
numpy_datas = numpy_all[0] if numpy_all else []
numpy_binaries = numpy_all[1] if len(numpy_all) > 1 else []

# Collect all data files
all_datas = list(torch_datas) + list(torchvision_datas) + list(numpy_datas) + [
    (str(PROJECT_ROOT / "configs"), "configs"),
    (str(PROJECT_ROOT / "src/picture_aliver"), "src/picture_aliver"),
    (str(PROJECT_ROOT / "src/utils"), "src/utils"),
    (str(PROJECT_ROOT / "src/core"), "src/core"),
    (str(PROJECT_ROOT / "src/modules"), "src/modules"),
]

# Collect numpy and other critical binaries
numpy_all = collect_all('numpy')
numpy_datas = numpy_all[0] if numpy_all else []
numpy_binaries = numpy_all[1] if len(numpy_all) > 1 else []

# Final binaries - include torch DLLs explicitly 
all_binaries = all_binaries + list(numpy_binaries)

a = Analysis(
    [str(SCRIPT_PATH)],
    pathex=[
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "src"),
    ],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=[
        # PyQt5
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        # PyTorch
        "torch",
        "torchvision",
        "torch.nn",
        "torch.nn.functional",
        "torch._C",
        # FastAPI
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.config",
        "fastapi",
        "starlette",
        "pydantic",
        "python_multipart",
        # Pipeline modules
        "src.picture_aliver.main",
        "src.picture_aliver.api",
        "src.picture_aliver.config",
        "src.picture_aliver.gpu_optimization",
        "src.picture_aliver.image_loader",
        "src.picture_aliver.depth_estimator",
        "src.picture_aliver.segmentation",
        "src.picture_aliver.motion_generator",
        "src.picture_aliver.video_generator",
        "src.picture_aliver.stabilizer",
        "src.picture_aliver.text_to_image",
        "src.picture_aliver.quality_control",
        "src.picture_aliver.exporter",
        "src.picture_aliver.model_manager",
        "src.picture_aliver.model_manager_extended",
# Core modules
        "src.core.pipeline",
        "src.core.model_registry",
        "src.core.model_loader",
        "src.core.device",
        "src.core.config",
        "src.core.config_extension",
        # Generation modules (includes furry models)
        "src.modules.generation.video_generator",
        "src.modules.generation.depth_conditioning",
        "src.modules.generation.furry_models",
        "src.modules.motion.motion_injector",
        "src.modules.motion.camera_motion",
        "src.modules.motion.furry_motions",
        "src.modules.depth.depth_estimator",
        "src.modules.segmentation.segmentor",
    ],
    hookspath=[],
    hooksconfig={},
    keys=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Console version (with terminal output)
exe_console = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Picture-Aliver-Console',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

# GUI version (no console)
exe_gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Picture-Aliver',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe_gui,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Picture-Aliver',
)