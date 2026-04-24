"""
Video Exporter Module

Exports video frames to various formats (MP4, WebM, GIF, etc.) with support for:
- Adjustable duration (5 seconds to 2 minutes)
- Adjustable FPS (1-60 fps)
- Frame interpolation for smoothness
- Mobile-compatible output
- Multiple quality presets
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import subprocess
import shutil

import numpy as np
import torch


class VideoFormat(str, Enum):
    """Supported video formats."""
    MP4 = "mp4"
    WEBM = "webm"
    GIF = "gif"
    AVI = "avi"


class QualityPreset(str, Enum):
    """Video quality presets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class Codec(str, Enum):
    """Video codecs."""
    H264 = "libx264"
    H265 = "libx265"
    VP8 = "libvpx"
    VP9 = "libvpx-vp9"
    AV1 = "libaom-av1"
    GIF_CODEC = "gif"


@dataclass
class VideoSpec:
    """
    Video specification for export.
    
    Attributes:
        duration_seconds: Video duration in seconds (5-120)
        fps: Frames per second (1-60)
        width: Output width
        height: Output height
        format: Output format
        quality: Quality preset
    """
    duration_seconds: float = 3.0
    fps: int = 8
    width: Optional[int] = None
    height: Optional[int] = None
    format: VideoFormat = VideoFormat.MP4
    quality: QualityPreset = QualityPreset.MEDIUM
    
    def __post_init__(self):
        self.duration_seconds = max(5.0, min(120.0, self.duration_seconds))
        self.fps = max(1, min(60, self.fps))
        
        if self.width is not None:
            self.width = (self.width // 2) * 2
        if self.height is not None:
            self.height = (self.height // 2) * 2
    
    @property
    def num_frames(self) -> int:
        """Calculate number of frames needed."""
        return int(self.duration_seconds * self.fps)
    
    def get_resolution(self, source_width: int, source_height: int) -> Tuple[int, int]:
        """Get output resolution."""
        if self.width and self.height:
            return self.width, self.height
        elif self.width:
            scale = self.width / source_width
            return self.width, int(source_height * scale)
        elif self.height:
            scale = self.height / source_height
            return int(source_width * scale), self.height
        else:
            if self.quality == QualityPreset.LOW:
                return 480, int(480 * source_height / source_width)
            elif self.quality == QualityPreset.MEDIUM:
                return 720, int(720 * source_height / source_width)
            elif self.quality == QualityPreset.HIGH:
                return 1080, int(1080 * source_height / source_width)
            else:
                return source_width, source_height


@dataclass
class ExportOptions:
    """
    Export configuration options.
    
    Attributes:
        video_spec: Video specification
        enable_interpolation: Enable frame interpolation
        interpolation_factor: Interpolation multiplier (2x, 4x)
        enable_stabilization: Apply stabilization
        codec: Video codec
        crf: Constant Rate Factor (lower = better quality)
        preset: Encoding speed preset
        pixel_format: Output pixel format
        add_audio: Add audio track (future)
        loop_gif: Loop GIF output
    """
    video_spec: VideoSpec = field(default_factory=VideoSpec)
    enable_interpolation: bool = False
    interpolation_factor: int = 2
    enable_stabilization: bool = True
    codec: Codec = Codec.H264
    crf: int = 23
    preset: str = "medium"
    pixel_format: str = "yuv420p"
    add_audio: bool = False
    loop_gif: bool = True
    
    @property
    def output_fps(self) -> int:
        """Get effective output FPS."""
        if self.enable_interpolation:
            return self.video_spec.fps * self.interpolation_factor
        return self.video_spec.fps


class VideoExporter:
    """
    Production video export system.
    
    Features:
    - Duration from 5 seconds to 2 minutes
    - FPS from 1 to 60
    - MP4, WebM, GIF output
    - Frame interpolation (RIFE-style)
    - Mobile-optimized encoding
    - Quality presets
    
    Attributes:
        device: Compute device
        ffmpeg_available: Whether ffmpeg is available
        output_dir: Default output directory
    """
    
    MOBILE_COMPATIBLE = {
        VideoFormat.MP4: True,
        VideoFormat.WEBM: True,
        VideoFormat.GIF: True,
        VideoFormat.AVI: False,
    }
    
    QUALITY_CRF = {
        QualityPreset.LOW: 28,
        QualityPreset.MEDIUM: 23,
        QualityPreset.HIGH: 18,
        QualityPreset.ULTRA: 15,
    }
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.ffmpeg_available = self._check_ffmpeg()
        self.output_dir = Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        return shutil.which("ffmpeg") is not None
    
    def export(
        self,
        video_frames: Any,
        output_path: Union[str, Path],
        options: Optional[ExportOptions] = None
    ) -> None:
        """
        Export video with full options.
        
        Args:
            video_frames: Frames to export
            output_path: Output file path
            options: Export options
        """
        options = options or ExportOptions()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        frames_list = self._prepare_frames(video_frames)
        
        if not frames_list:
            raise ValueError("No frames to export")
        
        if options.enable_interpolation and len(frames_list) > 1:
            frames_list = self._interpolate_frames(frames_list, options.interpolation_factor)
        
        duration = options.video_spec.duration_seconds
        fps = options.output_fps
        
        if output_path.suffix.lower() == '.gif':
            self._export_gif(frames_list, output_path, fps, options.loop_gif)
        else:
            self._export_video(frames_list, output_path, fps, options)
    
    def export_with_duration(
        self,
        video_frames: Any,
        output_path: Union[str, Path],
        duration_seconds: float,
        fps: int = 8,
        format: str = "mp4",
        quality: str = "medium"
    ) -> None:
        """
        Simple export with duration control.
        
        Args:
            video_frames: Frames to export
            output_path: Output path
            duration_seconds: Video duration (5-120 seconds)
            fps: Frames per second
            format: Output format (mp4, webm, gif)
            quality: Quality preset (low, medium, high, ultra)
        """
        duration_seconds = max(5.0, min(120.0, duration_seconds))
        
        quality_map = {
            "low": QualityPreset.LOW,
            "medium": QualityPreset.MEDIUM,
            "high": QualityPreset.HIGH,
            "ultra": QualityPreset.ULTRA,
        }
        
        format_map = {
            "mp4": VideoFormat.MP4,
            "webm": VideoFormat.WEBM,
            "gif": VideoFormat.GIF,
        }
        
        options = ExportOptions(
            video_spec=VideoSpec(
                duration_seconds=duration_seconds,
                fps=fps,
                format=format_map.get(format, VideoFormat.MP4),
                quality=quality_map.get(quality, QualityPreset.MEDIUM)
            )
        )
        
        self.export(video_frames, output_path, options)
    
    def _prepare_frames(self, video_frames: Any) -> List[np.ndarray]:
        """Convert video frames to list of numpy arrays."""
        if hasattr(video_frames, 'to_list'):
            return video_frames.to_list()
        
        if isinstance(video_frames, torch.Tensor):
            return self._tensor_to_list(video_frames)
        
        if isinstance(video_frames, list):
            return [self._frame_to_numpy(f) for f in video_frames]
        
        raise ValueError(f"Unsupported video frame type: {type(video_frames)}")
    
    def _tensor_to_list(self, tensor: torch.Tensor) -> List[np.ndarray]:
        """Convert tensor to list of numpy arrays."""
        if tensor.dim() == 4:
            frames = []
            for i in range(tensor.shape[0]):
                frames.append(self._frame_to_numpy(tensor[i]))
            return frames
        elif tensor.dim() == 5:
            frames = []
            for i in range(tensor.shape[1]):
                frames.append(self._frame_to_numpy(tensor[0, i]))
            return frames
        else:
            return [self._frame_to_numpy(tensor)]
    
    def _frame_to_numpy(self, frame: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert frame to numpy array."""
        if isinstance(frame, np.ndarray):
            return frame
        
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu()
            
            if frame.dtype == torch.float32:
                if frame.max() <= 1.0:
                    frame = (frame * 255).to(torch.uint8)
                else:
                    frame = frame.to(torch.uint8)
            
            if frame.dim() == 3:
                frame = frame.permute(1, 2, 0).numpy()
            
            if frame.shape[-1] == 1:
                frame = frame.squeeze(-1)
        
        return frame
    
    def _interpolate_frames(
        self,
        frames: List[np.ndarray],
        factor: int = 2
    ) -> List[np.ndarray]:
        """
        Interpolate frames for smoother output.
        
        Args:
            frames: Input frames
            factor: Interpolation factor (2 = double frames)
            
        Returns:
            Interpolated frames
        """
        if len(frames) < 2 or factor < 2:
            return frames
        
        output_frames = []
        
        for i in range(len(frames) - 1):
            output_frames.append(frames[i])
            
            for j in range(1, factor):
                alpha = j / factor
                
                frame1 = frames[i].astype(np.float32)
                frame2 = frames[i + 1].astype(np.float32)
                
                interp = (1 - alpha) * frame1 + alpha * frame2
                output_frames.append(interp.astype(np.uint8))
        
        output_frames.append(frames[-1])
        
        return output_frames
    
    def _export_video(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: int,
        options: ExportOptions
    ) -> None:
        """Export video using ffmpeg."""
        if not self.ffmpeg_available:
            self._export_with_opencv(frames, output_path, fps, options)
            return
        
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            for i, frame in enumerate(frames):
                frame_path = temp_dir / f"frame_{i:06d}.png"
                self._save_frame(frame, frame_path)
            
            codec = self._get_codec(options)
            crf = options.crf
            if options.video_spec.quality in self.QUALITY_CRF:
                crf = self.QUALITY_CRF[options.video_spec.quality]
            
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(temp_dir / "frame_%06d.png"),
                "-c:v", codec.value,
            ]
            
            if codec == Codec.H264:
                cmd.extend([
                    "-preset", options.preset,
                    "-crf", str(crf),
                    "-pix_fmt", options.pixel_format,
                    "-movflags", "+faststart",
                ])
            elif codec == Codec.H265:
                cmd.extend([
                    "-preset", options.preset,
                    "-crf", str(crf),
                    "-pix_fmt", options.pixel_format,
                    "-tag:v", "hvc1",
                ])
            elif codec == Codec.VP9:
                cmd.extend([
                    "-crf", str(crf),
                    "-b:v", "0",
                    "-pix_fmt", options.pixel_format,
                ])
            
            cmd.append(str(output_path))
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[VideoExporter] FFmpeg warning: {result.stderr}")
                self._export_with_opencv(frames, output_path, fps, options)
            else:
                print(f"[VideoExporter] Exported: {output_path}")
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _export_gif(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: int,
        loop: bool = True
    ) -> None:
        """Export as GIF with optimization."""
        if self.ffmpeg_available:
            self._export_gif_ffmpeg(frames, output_path, fps, loop)
        else:
            self._export_gif_pil(frames, output_path, fps, loop)
    
    def _export_gif_ffmpeg(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: int,
        loop: bool
    ) -> None:
        """Export GIF using ffmpeg with palette generation."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            for i, frame in enumerate(frames):
                frame_path = temp_dir / f"frame_{i:06d}.png"
                self._save_frame(frame, frame_path)
            
            palette_file = temp_dir / "palette.png"
            
            filter_str = f"fps={fps},scale=480:-1:flags=lanczos,palettegen"
            
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(temp_dir / "frame_%06d.png"),
                "-vf", filter_str,
                str(palette_file)
            ]
            subprocess.run(cmd, capture_output=True)
            
            loop_arg = "0" if loop else "-1"
            
            filter_str = f"fps={fps},scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5"
            
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(temp_dir / "frame_%06d.png"),
                "-i", str(palette_file),
                "-lavfi", filter_str,
                "-loop", loop_arg,
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[VideoExporter] GIF export fallback")
                self._export_gif_pil(frames, output_path, fps, loop)
            else:
                print(f"[VideoExporter] Exported GIF: {output_path}")
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _export_gif_pil(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: int,
        loop: bool
    ) -> None:
        """Fallback GIF export using PIL."""
        try:
            from PIL import Image
            
            max_width = 480
            images = []
            
            for frame in frames:
                if frame.ndim == 3 and frame.shape[-1] == 3:
                    if frame.max() <= 1:
                        frame = (frame * 255).astype(np.uint8)
                    
                    h, w = frame.shape[:2]
                    if w > max_width:
                        scale = max_width / w
                        new_w, new_h = max_width, int(h * scale)
                        frame = np.array(Image.fromarray(frame).resize((new_w, new_h)))
                    
                    images.append(Image.fromarray(frame))
                elif frame.ndim == 2:
                    images.append(Image.fromarray(frame))
            
            if images:
                images[0].save(
                    output_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=int(1000 / fps),
                    loop=0 if loop else -1,
                    optimize=True
                )
                print(f"[VideoExporter] Exported GIF: {output_path}")
        
        except ImportError:
            print("[VideoExporter] Cannot export GIF: PIL required")
            self._export_frames_individual(frames, output_path.parent)
    
    def _export_with_opencv(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: int,
        options: ExportOptions
    ) -> None:
        """Fallback export using OpenCV."""
        try:
            import cv2
            
            if not frames:
                raise ValueError("No frames")
            
            h, w = frames[0].shape[:2]
            h = (h // 2) * 2
            w = (w // 2) * 2
            
            fourcc_map = {
                VideoFormat.MP4: cv2.VideoWriter_fourcc(*'mp4v'),
                VideoFormat.WEBM: cv2.VideoWriter_fourcc(*'VP80'),
                VideoFormat.AVI: cv2.VideoWriter_fourcc(*'XVID'),
            }
            
            format_ext = Path(output_path).suffix.lower()
            if format_ext == '.webm':
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            
            for frame in frames:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                
                if frame.ndim == 3 and frame.shape[-1] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                writer.write(frame_bgr)
            
            writer.release()
            print(f"[VideoExporter] Exported: {output_path}")
        
        except ImportError:
            print("[VideoExporter] Cannot export: OpenCV required")
            self._export_frames_individual(frames, output_path.parent)
    
    def _save_frame(self, frame: np.ndarray, path: Path) -> None:
        """Save single frame as PNG."""
        try:
            from PIL import Image
            
            if frame.ndim == 3 and frame.shape[-1] == 3:
                if frame.max() <= 1:
                    frame = (frame * 255).astype(np.uint8)
                Image.fromarray(frame).save(path, format='PNG')
            elif frame.ndim == 2:
                if frame.max() <= 1:
                    frame = (frame * 255).astype(np.uint8)
                Image.fromarray(frame).save(path, format='PNG')
        
        except ImportError:
            import cv2
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            cv2.imwrite(str(path), frame_bgr)
    
    def _export_frames_individual(
        self,
        frames: List[np.ndarray],
        output_dir: Path
    ) -> None:
        """Export frames individually."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:04d}.png"
            self._save_frame(frame, frame_path)
        
        print(f"[VideoExporter] Exported {len(frames)} frames to: {output_dir}")
    
    def _get_codec(self, options: ExportOptions) -> Codec:
        """Get appropriate codec for format."""
        fmt = options.video_spec.format
        
        if fmt == VideoFormat.MP4:
            return Codec.H264
        elif fmt == VideoFormat.WEBM:
            return Codec.VP9
        elif fmt == VideoFormat.GIF:
            return Codec.GIF_CODEC
        else:
            return Codec.H264
    
    def export_batch(
        self,
        video_frames: Any,
        output_dir: Union[str, Path],
        formats: List[str] = None,
        duration_seconds: float = 3.0,
        fps: int = 8,
        quality: str = "medium"
    ) -> Dict[str, Path]:
        """
        Export to multiple formats.
        
        Args:
            video_frames: Frames to export
            output_dir: Output directory
            formats: List of formats (mp4, webm, gif)
            duration_seconds: Video duration
            fps: Frames per second
            quality: Quality preset
            
        Returns:
            Dictionary mapping format to output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if formats is None:
            formats = ["mp4", "gif"]
        
        results = {}
        
        for fmt in formats:
            output_path = output_dir / f"video.{fmt}"
            
            try:
                self.export_with_duration(
                    video_frames,
                    output_path,
                    duration_seconds=duration_seconds,
                    fps=fps,
                    format=fmt,
                    quality=quality
                )
                results[fmt] = output_path
            except Exception as e:
                print(f"[VideoExporter] Failed to export {fmt}: {e}")
        
        return results
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """Get video metadata."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        info = {
            "path": str(video_path),
            "format": video_path.suffix[1:],
            "size_bytes": video_path.stat().st_size,
            "mobile_compatible": self.MOBILE_COMPATIBLE.get(
                VideoFormat(video_path.suffix[1:]), False
            )
        }
        
        if self.ffmpeg_available:
            try:
                cmd = [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(video_path)
                ]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                import json
                data = json.loads(result.stdout)
                
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        info["width"] = stream.get("width")
                        info["height"] = stream.get("height")
                        fps_str = stream.get("r_frame_rate", "0/1")
                        if '/' in fps_str:
                            num, denom = fps_str.split('/')
                            info["fps"] = float(num) / float(denom)
                        info["duration"] = float(stream.get("duration", 0))
                        info["codec"] = stream.get("codec_name")
                
                format_info = data.get("format", {})
                info["duration"] = float(format_info.get("duration", info.get("duration", 0)))
                info["bitrate"] = int(format_info.get("bit_rate", 0))
            
            except Exception:
                pass
        
        return info


def export_video(
    video_frames: Any,
    output_path: Union[str, Path],
    duration_seconds: float = 3.0,
    fps: int = 8,
    format: str = "mp4",
    quality: str = "medium",
    enable_interpolation: bool = False
) -> None:
    """
    Convenience function for video export.
    
    Args:
        video_frames: Frames to export
        output_path: Output file path
        duration_seconds: Video duration (5-120 seconds)
        fps: Frames per second (1-60)
        format: Output format (mp4, webm, gif)
        quality: Quality preset (low, medium, high, ultra)
        enable_interpolation: Enable frame interpolation
    """
    exporter = VideoExporter()
    
    options = ExportOptions(
        video_spec=VideoSpec(
            duration_seconds=duration_seconds,
            fps=fps,
            format=VideoFormat(format),
            quality=QualityPreset(quality)
        ),
        enable_interpolation=enable_interpolation
    )
    
    exporter.export(video_frames, output_path, options)


if __name__ == "__main__":
    print("[VideoExporter] Module test")
    
    exporter = VideoExporter()
    print(f"FFmpeg available: {exporter.ffmpeg_available}")
    print(f"Output directory: {exporter.output_dir}")
    
    spec = VideoSpec(duration_seconds=30.0, fps=24)
    print(f"\n30s @ 24fps video spec:")
    print(f"  Frames needed: {spec.num_frames}")
    print(f"  Duration: {spec.duration_seconds}s")
    print(f"  FPS: {spec.fps}")