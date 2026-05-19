"""
Pic Aliver Debug System

Diagnostics, profiling, tracing, and logging for model inference.
"""

from __future__ import annotations

import os
import gc
import time
import json
import inspect
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from functools import wraps


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimingEvent:
    """A single timing measurement."""
    name: str
    start: float
    end: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"[{self.duration:.3f}s] {self.name}"


@dataclass
class VRAMSnapshot:
    """VRAM usage at a point in time."""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    total_mb: float

    @property
    def free_mb(self) -> float:
        return self.total_mb - self.allocated_mb

    def __repr__(self):
        return f"VRAM: {self.allocated_mb:.0f}/{self.total_mb:.0f}MB used"


@dataclass
class ModelLoadRecord:
    """Record of a model load event."""
    model_name: str
    pipeline_type: str
    repo_id: str
    load_time: float
    vram_before_mb: float
    vram_after_mb: float
    vram_increase_mb: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class GenerationTrace:
    """Full trace of a generation operation."""
    mode: str
    model_name: str
    prompt: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    total_vram_mb: float = 0
    peak_vram_mb: float = 0
    events: List[TimingEvent] = field(default_factory=list)
    model_loads: List[ModelLoadRecord] = field(default_factory=list)
    vram_snapshots: List[VRAMSnapshot] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, name: str, metadata: Optional[Dict] = None) -> TimingEvent:
        """Add a completed timing event."""
        now = time.time()
        if self.events and self.events[-1].name == name and self.events[-1].end == 0:
            evt = self.events[-1]
            evt.end = now
            evt.duration = evt.end - evt.start
            if metadata:
                evt.metadata.update(metadata)
            return evt
        evt = TimingEvent(name=name, start=now, end=0, duration=0, metadata=metadata or {})
        self.events.append(evt)
        return evt

    def end_event(self, name: str, metadata: Optional[Dict] = None) -> None:
        """End a previously started timing event."""
        now = time.time()
        for evt in reversed(self.events):
            if evt.name == name and evt.end == 0:
                evt.end = now
                evt.duration = evt.end - evt.start
                if metadata:
                    evt.metadata.update(metadata)
                return
        self.add_event(name, metadata)

    def snapshot_vram(self) -> VRAMSnapshot:
        """Take a VRAM snapshot."""
        snap = _vram_snapshot()
        self.vram_snapshots.append(snap)
        if snap.allocated_mb > self.peak_vram_mb:
            self.peak_vram_mb = snap.allocated_mb
        if snap.total_mb > self.total_vram_mb:
            self.total_vram_mb = snap.total_mb
        return snap

    def finish(self, success: bool, error: Optional[str] = None) -> None:
        """Mark the trace as complete."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error
        self.snapshot_vram()

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Generation: {self.model_name} [{self.mode}]",
            f"  Duration: {self.duration:.2f}s" if self.duration else "",
            f"  Prompt: {self.prompt[:80]}..." if len(self.prompt) > 80 else f"  Prompt: {self.prompt}",
            f"  Peak VRAM: {self.peak_vram_mb:.0f}MB",
            f"  Events: {len(self.events)}",
            f"  Success: {self.success}",
        ]
        if self.events:
            lines.append("  Timeline:")
            for evt in self.events:
                lines.append(f"    {evt}")
        if self.error:
            lines.append(f"  Error: {self.error}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "mode": self.mode,
            "model_name": self.model_name,
            "prompt": self.prompt[:200],
            "duration": round(self.duration, 3) if self.duration else None,
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "total_vram_mb": round(self.total_vram_mb, 1),
            "success": self.success,
            "error": self.error,
            "events": [
                {"name": e.name, "duration": round(e.duration, 3)}
                for e in self.events if e.duration
            ],
            "vram_snapshots": [
                {"allocated_mb": round(s.allocated_mb, 1), "free_mb": round(s.free_mb, 1)}
                for s in self.vram_snapshots
            ],
        }


# =============================================================================
# VRAM UTILITIES
# =============================================================================

def _vram_snapshot() -> VRAMSnapshot:
    """Take a VRAM snapshot."""
    if not _cuda_available():
        return VRAMSnapshot(
            timestamp=time.time(),
            allocated_mb=0, reserved_mb=0, peak_mb=0, total_mb=0,
        )
    import torch
    return VRAMSnapshot(
        timestamp=time.time(),
        allocated_mb=torch.cuda.memory_allocated() / (1024**2),
        reserved_mb=torch.cuda.memory_reserved() / (1024**2),
        peak_mb=torch.cuda.max_memory_allocated() / (1024**2),
        total_mb=torch.cuda.get_device_properties(0).total_memory / (1024**2),
    )


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# =============================================================================
# DEBUG MANAGER
# =============================================================================

class DebugManager:
    """
    Central debug manager for Pic Aliver models.
    
    Tracks model loads, generation traces, and provides diagnostics.
    """

    def __init__(self, enabled: bool = True, log_dir: Optional[Union[str, Path]] = None):
        self.enabled = enabled
        self.log_dir = Path(log_dir) if log_dir else Path("./debug_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._traces: List[GenerationTrace] = []
        self._model_loads: List[ModelLoadRecord] = []
        self._current_trace: Optional[GenerationTrace] = None
        self._performance_log: List[Dict] = []

    @property
    def last_trace(self) -> Optional[GenerationTrace]:
        return self._traces[-1] if self._traces else None

    def begin_trace(self, mode: str, model_name: str, prompt: str) -> GenerationTrace:
        """Start tracing a generation."""
        trace = GenerationTrace(
            mode=mode,
            model_name=model_name,
            prompt=prompt,
            start_time=time.time(),
        )
        trace.snapshot_vram()
        trace.add_event("total")
        self._current_trace = trace
        return trace

    def end_trace(self, success: bool = True, error: Optional[str] = None) -> GenerationTrace:
        """End the current trace."""
        if self._current_trace is None:
            raise RuntimeError("No trace in progress")
        trace = self._current_trace
        trace.end_event("total")
        trace.finish(success=success, error=error)
        self._traces.append(trace)
        self._performance_log.append(trace.to_dict())
        self._current_trace = None
        return trace

    def record_model_load(
        self,
        model_name: str,
        pipeline_type: str,
        repo_id: str,
        load_time: float,
        success: bool,
        error: Optional[str] = None,
    ) -> ModelLoadRecord:
        """Record a model load event."""
        before = _vram_snapshot()
        import torch
        if _cuda_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        after = _vram_snapshot()
        
        record = ModelLoadRecord(
            model_name=model_name,
            pipeline_type=pipeline_type,
            repo_id=repo_id,
            load_time=load_time,
            vram_before_mb=before.allocated_mb,
            vram_after_mb=after.allocated_mb,
            vram_increase_mb=after.allocated_mb - before.allocated_mb,
            success=success,
            error=error,
        )
        self._model_loads.append(record)
        return record

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get full system diagnostics."""
        diag = {
            "system": self._system_info(),
            "cuda": self._cuda_info(),
            "model_loads": len(self._model_loads),
            "total_traces": len(self._traces),
            "recent_traces": [
                t.to_dict() for t in self._traces[-5:]
            ] if self._traces else [],
            "recent_loads": [
                {
                    "model": r.model_name,
                    "time": round(r.load_time, 2),
                    "vram_mb": round(r.vram_increase_mb, 1),
                    "success": r.success,
                }
                for r in self._model_loads[-5:]
            ] if self._model_loads else [],
            "performance": self._performance_summary(),
        }
        return diag

    def _system_info(self) -> Dict[str, Any]:
        import platform
        try:
            import psutil
            mem = psutil.virtual_memory()
            ram = f"{mem.total / (1024**3):.1f}GB"
        except ImportError:
            ram = "N/A"
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ram": ram,
            "cpu_count": os.cpu_count(),
        }

    def _cuda_info(self) -> Dict[str, Any]:
        if not _cuda_available():
            return {"available": False}
        import torch
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "total_vram": f"{props.total_memory / (1024**3):.1f}GB",
            "compute_capability": f"{props.major}.{props.minor}",
            "cuda_version": torch.version.cuda,
        }

    def _performance_summary(self) -> Dict[str, Any]:
        if not self._performance_log:
            return {}
        durations = [p["duration"] for p in self._performance_log if p.get("duration")]
        vrams = [p["peak_vram_mb"] for p in self._performance_log if p.get("peak_vram_mb")]
        return {
            "avg_duration": round(sum(durations) / len(durations), 2) if durations else 0,
            "max_duration": round(max(durations), 2) if durations else 0,
            "total_generations": len(self._performance_log),
            "avg_peak_vram_mb": round(sum(vrams) / len(vrams), 1) if vrams else 0,
            "success_rate": f"{sum(1 for p in self._performance_log if p.get('success')) / len(self._performance_log) * 100:.0f}%",
        }

    def save_log(self) -> Path:
        """Save debug log to file."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"debug_{ts}.json"
        data = {
            "timestamp": ts,
            "diagnostics": self.get_diagnostics(),
            "all_model_loads": [
                asdict(r) for r in self._model_loads
            ],
            "all_traces": [t.to_dict() for t in self._traces],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def print_status(self) -> None:
        """Print debug status to console."""
        diag = self.get_diagnostics()
        print("\n" + "=" * 60)
        print("  FROSTING DEBUG DIAGNOSTICS")
        print("=" * 60)
        print(f"  System: {diag['system']['platform']}")
        print(f"  Python: {diag['system']['python']}")
        print(f"  RAM: {diag['system']['ram']}")
        if diag['cuda']['available']:
            print(f"  GPU: {diag['cuda']['device']}")
            print(f"  VRAM: {diag['cuda']['total_vram']}")
        else:
            print("  GPU: None (CPU mode)")
        print(f"\n  Model loads: {diag['model_loads']}")
        print(f"  Generations: {diag['total_traces']}")
        if diag.get('performance'):
            p = diag['performance']
            print(f"  Avg duration: {p.get('avg_duration', 'N/A')}s")
            print(f"  Success rate: {p.get('success_rate', 'N/A')}")
        print("=" * 60)


# =============================================================================
# DECORATOR FOR TRACING
# =============================================================================

def trace_generation(debug_manager: DebugManager):
    """Decorator to trace a generation function."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mode = kwargs.get("mode", "unknown")
            model_name = kwargs.get("model_name", "unknown")
            prompt = kwargs.get("prompt", "")
            trace = debug_manager.begin_trace(mode, model_name, prompt)
            try:
                result = func(*args, **kwargs)
                trace.add_event("inference_complete")
                debug_manager.end_trace(success=True)
                return result
            except Exception as e:
                debug_manager.end_trace(success=False, error=str(e))
                raise
        return wrapper
    return decorator


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

debug = DebugManager()
