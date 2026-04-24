# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller build specification for Picture-Aliver Desktop (PyQt5)

Build commands:
    pyinstaller desktop/pyqt/build.spec --noconfirm
    python -m PyInstaller desktop/pyqt/build.spec --noconfirm

Output:
    dist/Picture-Aliver/Picture-Aliver.exe
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Project root directory - use absolute path
project_root = Path("E:/Picture-Aliver")

a = Analysis(
    ['desktop/pyqt/main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / 'configs/default.yaml'), 'configs'),
        (str(project_root / 'src/picture_aliver/config.yaml'), 'src/picture_aliver'),
    ],
    hiddenimports=[
        # PyQt5
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        # PyTorch
        'torch',
        'torchvision',
        # FastAPI
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.config',
        'fastapi',
        'starlette',
        'pydantic',
        'python_multipart',
        # Pipeline
        'src.picture_aliver.main',
        'src.picture_aliver.api',
        'src.picture_aliver.config',
        'src.picture_aliver.gpu_optimization',
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