"""
Picture-Aliver Desktop App (PyQt5)

Full desktop application for AI image-to-video generation.
Includes integrated local backend server.

Run:
    python desktop/pyqt/main.py

Build:
    pyinstaller desktop/pyqt/build.spec
"""

import sys
import os
import threading
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path for both development and PyInstaller
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Handle PyInstaller bundled executable
if getattr(sys, 'frozen', False):
    _bundle_dir = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(sys.executable).parent
    if str(_bundle_dir) not in sys.path:
        sys.path.insert(0, str(_bundle_dir))
    # Add src module path for bundled app
    _src_path = _bundle_dir / "src"
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QSlider, QProgressBar,
    QFileDialog, QMessageBox, QComboBox, QGroupBox, QCheckBox,
    QStatusBar, QToolBar, QSplitter, QFrame, QScrollArea, QProgressDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QIcon, QFont, QPalette, QColor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('picture_aliver_desktop')


# =============================================================================
# BACKEND INTEGRATION
# =============================================================================

class LocalBackend:
    """
    Runs the FastAPI backend in a separate process or thread.
    Provides direct access to the pipeline without network overhead.
    """
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.process = None
        self.started = False
        
    def start(self) -> bool:
        """Start the local backend server."""
        if self.started:
            return True
            
        try:
            import subprocess
            import uvicorn
            
            # Get the api module path
            api_path = Path("src/picture_aliver/api.py").resolve()
            if not api_path.exists():
                api_path = Path(__file__).parent.parent.parent / "src/picture_aliver/api.py"
            
            # Start uvicorn in background thread
            def run_server():
                logger.info(f"Starting backend on port {self.port}...")
                sys.path.insert(0, str(api_path.parent.parent.parent))
                uvicorn.run(
                    "picture_aliver.api:app",
                    host="127.0.0.1",
                    port=self.port,
                    log_level="warning"
                )
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            # Wait for server to start
            time.sleep(2)
            self.started = True
            logger.info("Backend server started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start backend: {e}")
            return False
    
    def stop(self):
        """Stop the backend server."""
        self.started = False
        logger.info("Backend server stopped")
    
    def is_running(self) -> bool:
        """Check if backend is running."""
        return self.started


# =============================================================================
# GENERATION WORKER THREAD
# =============================================================================

class GenerationWorker(QThread):
    """Background thread for video generation."""
    
    progress = pyqtSignal(str, int)  # message, percentage
    finished = pyqtSignal(bool, str, str)  # success, message, video_path
    log_message = pyqtSignal(str)  # log line
    
    def __init__(self, image_path: str, params: dict, use_unified_model: bool = True, device: str = "cuda"):
        super().__init__()
        self.image_path = image_path
        self.params = params
        self.use_unified_model = use_unified_model
        self.device = device
        self._running = True
        
    def run(self):
        """Run video generation in background."""
        try:
            self.log_message.emit("[Worker] Starting generation...")
            self.progress.emit("Initializing pipeline...", 5)
            
            if self.use_unified_model:
                # Use new unified model interface
                self._run_unified()
            else:
                # Use legacy pipeline
                self._run_legacy()
                
        except Exception as e:
            self.log_message.emit(f"[Worker] Error: {e}")
            self.finished.emit(False, str(e), "")
    
    def _run_unified(self):
        """Run using unified model interface."""
        try:
            from src.picture_aliver.model_manager import ModelManager
            from datetime import datetime
            
            self.progress.emit("Loading model...", 10)
            self.log_message.emit("[Worker] Loading model with fallback support...")
            
            # Create manager with fallback and device
            manager = ModelManager(
                primary="wan21",
                fallback="legacy",
                device=self.device
            )
            
            self.progress.emit("Generating video...", 30)
            self.log_message.emit("[Worker] Starting video generation...")
            
            # Set output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f"video_{timestamp}.mp4")
            
            # Generate with output path
            result = manager.generate(
                image=self.image_path,
                prompt=self.params.get('prompt', ''),
                negative_prompt=self.params.get('negative_prompt', ''),
                duration=self.params.get('duration', 3.0),
                fps=self.params.get('fps', 8),
                width=self.params.get('width', 512),
                height=self.params.get('height', 512),
                output_path=output_path,
            )
            
            if result["success"]:
                self.progress.emit("Complete!", 100)
                self.log_message.emit(f"[Worker] Success! Video: {result['video_path']}")
                self.log_message.emit(f"[Worker] Model: {result['model_type']}, Time: {result['generation_time']:.1f}s")
                self.finished.emit(True, "Video generated successfully!", result["video_path"])
            else:
                error = result.get("error", "Unknown error")
                self.log_message.emit(f"[Worker] Failed: {error}")
                # Show which models were tried
                if "attempts" in result:
                    for attempt in result["attempts"]:
                        model_name = attempt.get("model_type", "unknown")
                        model_error = attempt.get("error", "unknown")
                        self.log_message.emit(f"  -> {model_name}: {model_error}")
                self.finished.emit(False, f"Generation failed: {error}", "")
                
        except ImportError as e:
            self.log_message.emit(f"[Worker] Using legacy mode (unified model unavailable): {e}")
            self._run_legacy()
        except Exception as e:
            self.log_message.emit(f"[Worker] Unified model error: {e}, falling back to legacy")
            try:
                self._run_legacy()
            except Exception as e2:
                self.log_message.emit(f"[Worker] Legacy failed too: {e2}")
                self.finished.emit(False, str(e2), "")
    
    def _run_legacy(self):
        """Run using legacy pipeline."""
        from src.picture_aliver.main import Pipeline, PipelineConfig, DebugConfig
        config = PipelineConfig(
            duration_seconds=self.params.get('duration', 3.0),
            fps=self.params.get('fps', 8),
            width=self.params.get('width', 512),
            height=self.params.get('height', 512),
            guidance_scale=self.params.get('guidance_scale', 7.5),
            motion_strength=self.params.get('motion_strength', 0.8),
            motion_mode=self.params.get('motion_mode', 'auto'),
            enable_quality_check=self.params.get('enable_quality_check', True),
            enable_stabilization=True,
            device=self.device,
            debug=DebugConfig(enabled=False)
        )
        
        self.progress.emit("Loading models...", 15)
        pipeline = Pipeline(config)
        pipeline.initialize()
        
        self.progress.emit("Running image-to-video generation...", 30)
        self.log_message.emit("[Worker] Pipeline initialized, starting generation...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"video_{timestamp}.mp4"
        
        result = pipeline.run_pipeline(
            image_path=self.image_path,
            prompt=self.params.get('prompt', ''),
            config=config,
            output_path=output_path
        )
        
        if result.success:
            self.progress.emit("Complete!", 100)
            self.log_message.emit("[Worker] Success! Video saved to " + str(result.output_path))
            self.finished.emit(True, "Video generated successfully!", str(result.output_path))
        else:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            self.log_message.emit("[Worker] Failed: " + errors)
            self.finished.emit(False, "Generation failed: " + errors, "")


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.worker: Optional[GenerationWorker] = None
        self.selected_image_path: Optional[str] = None
        self._use_unified_model = True
        self.last_video_path: Optional[str] = None
        
        self.init_ui()
        self.setup_signals()
        
        # Check GPU status
        self.check_gpu_status()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Picture-Aliver Desktop")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QHBoxLayout(central)
        
        # Left panel - Input
        left_panel = self.create_input_panel()
        
        # Right panel - Output/Preview
        right_panel = self.create_output_panel()
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("✅ Ready - Select an image and enter a prompt to generate video")
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: #10B981;
                color: white;
                padding: 4px;
            }
        """)
        
        # Menu bar
        self.create_menu_bar()
        
        # Toolbar
        self.create_toolbar()
    
    def create_input_panel(self) -> QWidget:
        """Create the left input panel with scroll support."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                width: 10px;
                background: #f1f1f1;
            }
            QScrollBar::handle:vertical {
                background: #c1c1c1;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a1a1a1;
            }
        """)
        
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title with logo
        title_layout = QHBoxLayout()
        title = QLabel("Picture-Aliver")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: #6366F1;")
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        subtitle = QLabel("AI Image-to-Video")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #888;")
        title_layout.addWidget(subtitle)
        
        layout.addLayout(title_layout)
        
        # Image selection
        image_group = QGroupBox("Source Image")
        image_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        image_layout = QVBoxLayout(image_group)
        
        self.image_label = QLabel("No image selected\n\nClick 'Select Image' to choose an image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #6366F1;
                border-radius: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                color: #6c757d;
                padding: 20px;
            }
        """)
        image_layout.addWidget(self.image_label)
        
        btn_layout = QHBoxLayout()
        self.select_btn = QPushButton("📁 Select Image")
        self.select_btn.setIcon(QIcon.fromTheme("document-open"))
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #e9ecef;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #dee2e6; }
        """)
        self.select_btn.clicked.connect(self.select_image)
        self.clear_btn = QPushButton("🗑 Clear")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #e9ecef;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #dee2e6; }
        """)
        self.clear_btn.clicked.connect(self.clear_image)
        btn_layout.addWidget(self.select_btn)
        btn_layout.addWidget(self.clear_btn)
        image_layout.addLayout(btn_layout)
        
        layout.addWidget(image_group)
        
        # Prompt input
        prompt_group = QGroupBox("Animation Prompt")
        prompt_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        prompt_layout = QVBoxLayout(prompt_group)
        
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the motion you want...\n\nExamples:\n• Gentle wave motion\n• Wind blowing through trees\n• Cinematic pan left to right\n• Subtle breathing animation")
        self.prompt_edit.setMaximumHeight(120)
        self.prompt_edit.setStyleSheet("""
            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                background: white;
                font-size: 13px;
            }
            QTextEdit:focus {
                border: 2px solid #6366F1;
            }
        """)
        prompt_layout.addWidget(self.prompt_edit)
        
        layout.addWidget(prompt_group)
        
        # Model Selection
        model_group = QGroupBox("Model Selection")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        model_layout = QVBoxLayout(model_group)
        
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border-radius: 6px;
                border: 1px solid #dee2e6;
            }
        """)
        model_select_layout.addWidget(self.model_combo)
        model_select_layout.addStretch()
        model_layout.addLayout(model_select_layout)
        
        # Add default items initially, will be refreshed on startup
        self.model_combo.addItems([
            "Loading models...",
        ])
        
        # Model info
        self.model_info = QLabel("VRAM: Checking...")
        self.model_info.setStyleSheet("color: #6c757d; font-size: 11px;")
        model_layout.addWidget(self.model_info)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Models")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #f3f4f6;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #e5e7eb; }
        """)
        refresh_btn.clicked.connect(self.refresh_model_list)
        model_layout.addWidget(refresh_btn)
        
        layout.addWidget(model_group)
        
        # Settings
        settings_group = QGroupBox("Generation Settings")
        settings_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        settings_layout = QVBoxLayout(settings_group)
        
        # Duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("⏱ Duration:"))
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setMinimum(1)
        self.duration_slider.setMaximum(30)
        self.duration_slider.setValue(3)
        self.duration_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                margin: -5px 0;
                background: #6366F1;
                border-radius: 9px;
            }
        """)
        self.duration_label = QLabel("3s")
        self.duration_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.duration_slider.valueChanged.connect(
            lambda v: self.duration_label.setText(f"{v}s")
        )
        duration_layout.addWidget(self.duration_slider)
        duration_layout.addWidget(self.duration_label)
        settings_layout.addLayout(duration_layout)
        
        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("🎬 FPS:"))
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["4", "6", "8", "12", "16", "24", "30"])
        self.fps_combo.setCurrentText("8")
        self.fps_combo.setStyleSheet("padding: 6px; border-radius: 4px;")
        fps_layout.addWidget(self.fps_combo)
        fps_layout.addStretch()
        
        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("📐 Resolution:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(["256", "384", "512", "768", "1024"])
        self.res_combo.setCurrentText("512")
        self.res_combo.setStyleSheet("padding: 6px; border-radius: 4px;")
        res_layout.addWidget(self.res_combo)
        res_layout.addStretch()
        settings_layout.addLayout(fps_layout)
        settings_layout.addLayout(res_layout)
        
        # Motion mode
        motion_layout = QHBoxLayout()
        motion_layout.addWidget(QLabel("🌊 Motion:"))
        self.motion_combo = QComboBox()
        self.motion_combo.addItems(["auto", "subtle", "cinematic", "zoom", "pan", "furry"])
        self.motion_combo.setCurrentText("auto")
        self.motion_combo.setStyleSheet("padding: 6px; border-radius: 4px;")
        motion_layout.addWidget(self.motion_combo)
        motion_layout.addStretch()
        settings_layout.addLayout(motion_layout)
        
        # Quality check
        self.quality_check = QCheckBox("✨ Enable AI auto-correction")
        self.quality_check.setChecked(True)
        self.quality_check.setStyleSheet("font-weight: bold;")
        settings_layout.addWidget(self.quality_check)
        
        layout.addWidget(settings_group)
        
        # Device selection (GPU/CPU)
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("⚡ Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.setCurrentText("cuda")
        self.device_combo.setStyleSheet("padding: 6px; border-radius: 4px; font-weight: bold;")
        self.device_combo.setToolTip("GPU requires CUDA-compatible graphics card")
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        layout.addLayout(device_layout)

        # Generate button
        self.generate_btn = QPushButton("🎬 Generate Video")
        self.generate_btn.setMinimumHeight(55)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366F1, stop:1 #8B5CF6);
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                padding: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4F46E5, stop:1 #7C3AED);
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #888;
            }
        """)
        self.generate_btn.clicked.connect(self.start_generation)
        layout.addWidget(self.generate_btn)
        
        layout.addStretch()
        scroll.setWidget(panel)
        return scroll
    
    def create_output_panel(self) -> QWidget:
        """Create the right output panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Output & Preview")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #6366F1;")
        layout.addWidget(title)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 6px;
                height: 12px;
                background: #e9ecef;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366F1, stop:1 #8B5CF6);
                border-radius: 6px;
            }
        """)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #6c757d; font-weight: bold;")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        
        # Video preview
        preview_group = QGroupBox("Generated Video")
        preview_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        preview_layout = QVBoxLayout(preview_group)
        
        self.video_label = QLabel("No video generated yet\n\nGenerate a video to see preview here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(250)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #dee2e6;
                border-radius: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                color: #6c757d;
                padding: 20px;
            }
        """)
        preview_layout.addWidget(self.video_label)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #059669; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.play_btn.clicked.connect(self.play_video)
        
        self.save_btn = QPushButton("💾 Save As...")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2563EB; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.save_btn.clicked.connect(self.save_video)
        
        btn_layout.addWidget(self.play_btn)
        btn_layout.addWidget(self.save_btn)
        preview_layout.addLayout(btn_layout)
        
        layout.addWidget(preview_group)
        
        # Log output
        log_group = QGroupBox("Processing Log")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #00ff00;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        # Clear log button
        clear_log_btn = QPushButton("🗑 Clear Log")
        clear_log_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 6px 12px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #5a6268; }
        """)
        clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        log_layout.addWidget(clear_log_btn)
        
        layout.addWidget(log_group)
        
        # Status info
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("🎯 GPU:"))
        self.gpu_label = QLabel("Checking...")
        self.gpu_label.setStyleSheet("color: #10B981; font-weight: bold;")
        status_layout.addWidget(self.gpu_label)
        status_layout.addStretch()
        status_layout.addWidget(QLabel("💾 VRAM:"))
        self.vram_label = QLabel("...")
        self.vram_label.setStyleSheet("color: #8B5CF6; font-weight: bold;")
        status_layout.addWidget(self.vram_label)
        layout.addLayout(status_layout)
        
        return panel
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background: #f8f9fa;
                padding: 4px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background: #e9ecef;
            }
            QMenu {
                background: white;
                border: 1px solid #dee2e6;
            }
            QMenu::item:selected {
                background: #6366F1;
                color: white;
            }
        """)
        
        # File menu
        file_menu = menubar.addMenu("📁 File")
        file_menu.addAction("📂 Open Image...", self.select_image, "Ctrl+O")
        file_menu.addAction("💾 Save Video As...", self.save_video, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("🚪 Exit", self.close, "Alt+F4")
        
        # Generate menu
        gen_menu = menubar.addMenu("🎬 Generate")
        gen_menu.addAction("▶ Start Generation", self.start_generation, "Ctrl+G")
        gen_menu.addAction("⏹ Stop Generation", self.stop_generation, "Ctrl+.")
        gen_menu.addSeparator()
        gen_menu.addAction("🗑 Clear Image", self.clear_image)
        
        # Settings menu
        settings_menu = menubar.addMenu("⚙ Settings")
        settings_menu.addAction("🎨 Preferences", self.show_settings)
        settings_menu.addAction("🔧 Backend Config", self.show_backend_config)
        
        # Help menu
        help_menu = menubar.addMenu("❓ Help")
        help_menu.addAction("📖 Documentation", self.show_docs)
        help_menu.addAction("ℹ About", self.show_about)
    
    def create_toolbar(self):
        """Create the application toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background: #f8f9fa;
                padding: 4px;
                spacing: 8px;
            }
            QToolButton {
                padding: 8px 12px;
                border-radius: 6px;
            }
            QToolButton:hover {
                background: #e9ecef;
            }
        """)
        self.addToolBar(toolbar)
        
        toolbar.addAction("📂 Open", self.select_image)
        toolbar.addSeparator()
        toolbar.addAction("🎬 Generate", self.start_generation)
        toolbar.addAction("⏹ Stop", self.stop_generation)
        toolbar.addSeparator()
        toolbar.addAction("💾 Save", self.save_video)
    
    def setup_signals(self):
        """Connect signals and slots."""
        pass
    
    def select_image(self):
        """Open file dialog to select image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        
        if file_path:
            self.selected_image_path = file_path
            pixmap = QPixmap(file_path)
            scaled = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            self.image_label.setText("")
            self.log(f"Selected image: {file_path}")
    
    def clear_image(self):
        """Clear selected image."""
        self.selected_image_path = None
        self.image_label.clear()
        self.image_label.setText("No image selected")
    
    def start_generation(self):
        """Start video generation."""
        if not self.selected_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first!")
            return
        
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a prompt!")
            return
        
        # Get selected model from userData
        selected_model = self.model_combo.currentData() or "legacy"
        if selected_model == "Loading models...":
            selected_model = "legacy"
        
# Get settings
        device = self.device_combo.currentText()
        params = {
            'duration': self.duration_slider.value(),
            'fps': int(self.fps_combo.currentText()),
            'width': int(self.res_combo.currentText()),
            'height': int(self.res_combo.currentText()),
            'guidance_scale': 7.5,
            'motion_strength': 0.8,
            'motion_mode': self.motion_combo.currentText(),
            'enable_quality_check': self.quality_check.isChecked(),
            'prompt': prompt,
            'model': selected_model,
            'device': device,
        }
        
        # Start worker thread with unified model
        self._use_unified_model = (selected_model != "legacy")
        self.worker = GenerationWorker(self.selected_image_path, params, self._use_unified_model, device)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.log_message.connect(self.log)
        self.worker.start()
    
    def stop_generation(self):
        """Stop current generation."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.log("Generation cancelled by user")
            self.generate_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    @pyqtSlot(str, int)
    def on_progress(self, message: str, percentage: int):
        """Handle progress updates."""
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(message)
        self.status_bar.showMessage(f"⏳ {message}")
    
    @pyqtSlot(bool, str, str)
    def on_finished(self, success: bool, message: str, video_path: str):
        """Handle generation completion."""
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_bar.showMessage("✅ Generation complete!")
            self.status_bar.setStyleSheet("""
                QStatusBar {
                    background: #10B981;
                    color: white;
                    padding: 4px;
                }
            """)
            self.log("=" * 40)
            self.log("✅ SUCCESS!")
            self.log(f"📁 Video: {video_path}")
            self.log("=" * 40)
            self.last_video_path = video_path
            self.play_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.video_label.setText(f"🎬 Video Generated!\n\n{video_path}")
            self.video_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #10B981;
                    border-radius: 12px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #d1fae5, stop:1 #a7f3d0);
                    color: #065f46;
                    padding: 20px;
                    font-weight: bold;
                }
            """)
            QMessageBox.information(self, "✅ Success", f"Video generated successfully!\n\n{video_path}")
        else:
            self.status_bar.showMessage("❌ Generation failed!")
            self.status_bar.setStyleSheet("""
                QStatusBar {
                    background: #EF4444;
                    color: white;
                    padding: 4px;
                }
            """)
            self.log("=" * 40)
            self.log("❌ FAILED!")
            self.log(f"Error: {message}")
            self.log("=" * 40)
            QMessageBox.critical(self, "❌ Error", f"Generation failed:\n\n{message}")
    
    def play_video(self):
        """Play the generated video."""
        if hasattr(self, 'last_video_path') and self.last_video_path:
            import subprocess
            import platform
            if platform.system() == 'Windows':
                os.startfile(self.last_video_path)
            else:
                subprocess.run(['xdg-open', self.last_video_path])
    
    def save_video(self):
        """Save video to custom location."""
        if hasattr(self, 'last_video_path') and self.last_video_path:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Video", "", "MP4 (*.mp4)"
            )
            if save_path:
                import shutil
                shutil.copy(self.last_video_path, save_path)
                self.log(f"Saved to: {save_path}")
    
    def show_settings(self):
        """Show settings dialog."""
        QMessageBox.information(self, "Settings", "Settings dialog coming soon")
    
    def show_backend_config(self):
        """Show backend configuration dialog."""
        QMessageBox.information(self, "Backend", f"Local backend running on port 8000")
    
    def show_docs(self):
        """Show documentation."""
        QMessageBox.information(self, "Documentation", 
            "Picture-Aliver Desktop\n\n"
            "1. Select an image\n"
            "2. Enter animation prompt\n"
            "3. Adjust settings\n"
            "4. Click Generate Video\n\n"
            "For more help, see README.md"
        )
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About Picture-Aliver",
            "Picture-Aliver Desktop v1.0.0\n\n"
            "AI Image-to-Video Generation Pipeline\n\n"
            "Built with PyQt5 and PyTorch"
        )
    
    def log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def check_gpu_status(self):
        """Check and display GPU status."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.gpu_label.setText(gpu_name)
                self.vram_label.setText(f"{vram:.1f} GB")
                self.model_info.setText(f"✓ GPU: {gpu_name} ({vram:.1f}GB VRAM)")
            else:
                self.gpu_label.setText("CPU")
                self.vram_label.setText("N/A")
                self.model_info.setText("⚠ CPU mode - GPU recommended for speed")
        except Exception as e:
            self.gpu_label.setText("Unknown")
            self.vram_label.setText("?")
            self.model_info.setText(f"Error checking GPU: {str(e)[:30]}")
        
        self.refresh_model_list()
    
    def get_available_models(self):
        """Detect available models from MODEL_REGISTRY."""
        available = []
        
        logger.info("[Model Discovery] Scanning model registry...")
        
        try:
            from src.core.model_registry import MODEL_REGISTRY, ModelCategory, ContentRating
            
            # Get all I2V models from registry
            i2v_models = MODEL_REGISTRY.get_by_category(ModelCategory.I2V)
            
            # Sort by rating (nsfw first, then mature, then safe) for easier access
            rating_order = {"nsfw": 0, "mature": 1, "safe": 2}
            i2v_models = sorted(i2v_models, key=lambda m: (rating_order.get(m.rating.value, 1), m.name))
            
            for model in i2v_models:
                model_id = model.model_path.lower()
                rating_icon = {"nsfw": "[+]", "mature": "[~]", "safe": "[-]"}[model.rating.value]
                display_name = f"{model.name} {rating_icon}"
                available.append((model_id, display_name))
                logger.info(f"[Model Discovery] {model.name} [{model.rating.value}]")
            
            # Add legacy pipeline
            available.append(("legacy", "Legacy Pipeline"))
            
            logger.info(f"[Model Discovery] Found {len(i2v_models)} I2V models")
            
        except ImportError as e:
            logger.warning(f"[Model Discovery] Cannot import MODEL_REGISTRY: {e}")
            available = self._get_default_models()
        
        # If still empty, use defaults
        if not available:
            available = self._get_default_models()
        
        logger.info(f"[Model Discovery] Total models: {len(available)}")
        return available
    
    def _get_default_models(self):
        """Get default model list."""
        return [
            ("wan21", "Wan 2.1 (High Quality)"),
            ("wan22", "Wan 2.2 (Latest)"),
            ("lightx2v", "LightX2V (Fast)"),
            ("svd", "SVD (Stable Video) [-]"),
            ("svd_xt", "SVD-XT [-]"),
            ("zeroscope", "ZeroScope [-]"),
            ("i2vgen_xl", "I2VGen-XL [-]"),
            ("hunyuan", "HunyuanVideo [-]"),
            ("ltx", "LTX-Video [-]"),
            ("cogvideo", "CogVideo [-]"),
            ("fluffyrock", "Fluffyrock [+]"),
            ("fluffyrock_unbound", "Fluffyrock-Unbound [+]"),
            ("yiffymix", "Yiffymix [+]"),
            ("yiffymix_v2", "Yiffymix-V2 [+]"),
            ("dreamshaper", "Dreamshaper [~]"),
            ("dreamshaper_xl", "Dreamshaper-XL [~]"),
            ("pawpunk", "PawPunk [+]"),
            ("furryforge", "FurryForge [+]"),
            ("pony_diffusion", "Pony-Diffusion [+]"),
            ("animatediff", "AnimateDiff-v3 [+]"),
            ("animatediff_sdxl", "AnimateDiff-SDXL [+]"),
            ("svd_open", "Open-SVD [+]"),
            ("zeroscope_open", "ZeroScope Unrestricted [+]"),
            ("legacy", "Legacy Pipeline"),
        ]
    
    def _check_model_available(self, model_type: str) -> bool:
        """Check if a specific model type is available."""
        try:
            if model_type == "legacy":
                # Legacy pipeline check - requires basic imports
                from src.picture_aliver.main import Pipeline
                return True
            
            # Check diffusers-based models
            if model_type in ("wan21", "wan22"):
                try:
                    from diffusers import WanImageToVideoPipeline
                    return True
                except ImportError:
                    pass
            
            # Check LightX2V
            if model_type == "lightx2v":
                try:
                    from lightx2v import LightX2VPipeline
                    return True
                except ImportError:
                    pass
            
            # Check HunyuanVideo
            if model_type == "hunyuan":
                try:
                    from diffusers import HunyuanVideoPipeline
                    return True
                except ImportError:
                    pass
            
            # Check LTX-Video
            if model_type == "ltx":
                try:
                    from diffusers import LTXVideoPipeline
                    return True
                except ImportError:
                    pass
            
            # CogVideo, LongCat don't need specific imports check
            if model_type in ("cogvideo", "longcat"):
                return True
            
        except Exception as e:
            logger.debug(f"[Model Discovery] {model_type} check failed: {e}")
        
        return False
    
    def refresh_model_list(self):
        """Refresh the model dropdown with available models."""
        try:
            self.model_combo.clear()
            available_models = self.get_available_models()
            
            for model_id, model_name in available_models:
                self.model_combo.addItem(model_name, userData=model_id)
            
            # Log detected models
            detected = [name for _, name in available_models]
            logger.info(f"[UI] Model dropdown populated with: {detected}")
            self.log(f"[Model Discovery] Found {len(available_models)} model(s)")
            
        except Exception as e:
            logger.error(f"[UI] Failed to refresh model list: {e}")
            # Fallback to default list
            self.model_combo.clear()
            self.model_combo.addItems([
                "Auto (Best Available)",
                "Wan 2.1 (High Quality)",
                "LightX2V (Fast)",
                "Legacy Pipeline"
            ])
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        event.accept()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application-wide stylesheet
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()