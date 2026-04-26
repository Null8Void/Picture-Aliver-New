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

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
    
    def __init__(self, image_path: str, params: dict, use_unified_model: bool = True):
        super().__init__()
        self.image_path = image_path
        self.params = params
        self.use_unified_model = use_unified_model
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
            
            self.progress.emit("Loading model...", 10)
            self.log_message.emit("[Worker] Loading model with fallback support...")
            
            # Create manager with fallback
            manager = ModelManager(
                primary="wan21",
                fallback="legacy"
            )
            
            self.progress.emit("Generating video...", 30)
            self.log_message.emit("[Worker] Starting video generation...")
            
            # Generate
            result = manager.generate(
                image=self.image_path,
                prompt=self.params.get('prompt', ''),
                negative_prompt=self.params.get('negative_prompt', ''),
                duration=self.params.get('duration', 3.0),
                fps=self.params.get('fps', 8),
                width=self.params.get('width', 512),
                height=self.params.get('height', 512),
            )
            
            if result["success"]:
                self.progress.emit("Complete!", 100)
                self.log_message.emit(f"[Worker] Success! Video: {result['video_path']}")
                self.log_message.emit(f"[Worker] Model: {result['model_type']}, Time: {result['generation_time']:.1f}s")
                self.finished.emit(True, "Video generated successfully!", result["video_path"])
            else:
                error = result.get("error", "Unknown error")
                self.log_message.emit(f"[Worker] Failed: {error}")
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
        """Request worker to stop."""
        self._running = False


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
        
        self.init_ui()
        self.setup_signals()
        
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
        self.status_bar.showMessage("Ready")
        
        # Menu bar
        self.create_menu_bar()
        
        # Toolbar
        self.create_toolbar()
    
    def create_input_panel(self) -> QWidget:
        """Create the left input panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Image-to-Video Generation")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Image selection
        image_group = QGroupBox("Source Image")
        image_layout = QVBoxLayout(image_group)
        
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet("border: 2px dashed #ccc; border-radius: 8px; background: #f5f5f5;")
        image_layout.addWidget(self.image_label)
        
        btn_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select Image")
        self.select_btn.clicked.connect(self.select_image)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_image)
        btn_layout.addWidget(self.select_btn)
        btn_layout.addWidget(self.clear_btn)
        image_layout.addLayout(btn_layout)
        
        layout.addWidget(image_group)
        
        # Prompt input
        prompt_group = QGroupBox("Animation Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the motion you want (e.g., gentle wave, wind blowing, cinematic pan)")
        self.prompt_edit.setMaximumHeight(100)
        prompt_layout.addWidget(self.prompt_edit)
        
        layout.addWidget(prompt_group)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration:"))
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setMinimum(1)
        self.duration_slider.setMaximum(30)
        self.duration_slider.setValue(3)
        self.duration_label = QLabel("3s")
        self.duration_slider.valueChanged.connect(
            lambda v: self.duration_label.setText(f"{v}s")
        )
        duration_layout.addWidget(self.duration_slider)
        duration_layout.addWidget(self.duration_label)
        settings_layout.addLayout(duration_layout)
        
        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["4", "6", "8", "12", "16", "24", "30"])
        self.fps_combo.setCurrentText("8")
        fps_layout.addWidget(self.fps_combo)
        fps_layout.addStretch()
        settings_layout.addLayout(fps_layout)
        
        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(["256", "384", "512", "768", "1024"])
        self.res_combo.setCurrentText("512")
        res_layout.addWidget(self.res_combo)
        res_layout.addStretch()
        settings_layout.addLayout(res_layout)
        
        # Motion mode
        motion_layout = QHBoxLayout()
        motion_layout.addWidget(QLabel("Motion:"))
        self.motion_combo = QComboBox()
        self.motion_combo.addItems(["auto", "subtle", "cinematic", "zoom", "pan", "furry"])
        self.motion_combo.setCurrentText("auto")
        motion_layout.addWidget(self.motion_combo)
        motion_layout.addStretch()
        settings_layout.addLayout(motion_layout)
        
        # Quality check
        self.quality_check = QCheckBox("Enable auto-correction")
        self.quality_check.setChecked(True)
        settings_layout.addWidget(self.quality_check)
        
        layout.addWidget(settings_group)
        
        # Generate button
        self.generate_btn = QPushButton("Generate Video")
        self.generate_btn.setMinimumHeight(50)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366F1;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #4F46E5; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.generate_btn.clicked.connect(self.start_generation)
        layout.addWidget(self.generate_btn)
        
        layout.addStretch()
        return panel
    
    def create_output_panel(self) -> QWidget:
        """Create the right output panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Output")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        
        # Video preview
        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.video_label = QLabel("No video generated yet")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(300)
        self.video_label.setStyleSheet("border: 2px solid #ccc; border-radius: 8px; background: #f5f5f5;")
        preview_layout.addWidget(self.video_label)
        
        self.play_btn = QPushButton("Play Video")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_video)
        preview_layout.addWidget(self.play_btn)
        
        layout.addWidget(preview_group)
        
        # Log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return panel
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open Image", self.select_image, "Ctrl+O")
        file_menu.addAction("Save Video", self.save_video, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, "Ctrl+Q")
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        settings_menu.addAction("Preferences", self.show_settings)
        settings_menu.addAction("Backend Config", self.show_backend_config)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Documentation", self.show_docs)
        help_menu.addAction("About", self.show_about)
    
    def create_toolbar(self):
        """Create the application toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        toolbar.addAction("Open", self.select_image)
        toolbar.addAction("Generate", self.start_generation)
        toolbar.addSeparator()
        toolbar.addAction("Stop", self.stop_generation)
    
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
        
        # Get settings
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
        }
        
        # Update UI
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Generating...")
        self.log(f"Starting generation with: {params}")
        
        # Start worker thread with unified model
        self.worker = GenerationWorker(self.selected_image_path, params, self._use_unified_model)
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
        self.status_bar.showMessage(message)
    
    @pyqtSlot(bool, str, str)
    def on_finished(self, success: bool, message: str, video_path: str):
        """Handle generation completion."""
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_bar.showMessage("Generation complete!")
            self.log(f"SUCCESS: {message}")
            self.last_video_path = video_path
            self.play_btn.setEnabled(True)
            self.video_label.setText(f"Video ready:\n{video_path}")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_bar.showMessage("Generation failed!")
            self.log(f"FAILED: {message}")
            QMessageBox.critical(self, "Error", message)
    
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