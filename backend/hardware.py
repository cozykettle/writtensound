"""Cheap startup hardware probe. Never makes network calls."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from .catalog import module_available
from .config import EnvConfig


class DeviceInfo(BaseModel):
    platform: str
    slug: str


class HardwareStatus(BaseModel):
    image_variant: str
    devices: dict[str, DeviceInfo] = Field(default_factory=dict)
    cpu_cores: int
    ram_gb: float
    vulkan_ok: bool
    cuda_ok: bool
    notes: list[str] = Field(default_factory=list)


def _ram_gb() -> float:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            windll = getattr(ctypes, "windll", None)
            if windll is not None and windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(round(status.ullTotalPhys / 1024**3, 1))
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except (OSError, ValueError, AttributeError):
        pass
    return 0.0


def _cpu_name() -> str:
    name = platform.processor() or platform.machine() or "CPU"
    return name.strip() or "CPU"


def _cuda_available(image_variant: str, notes: list[str]) -> bool:
    if image_variant != "cuda":
        notes.append("Standard image: CUDA is unavailable by design. Use the :cuda image for NVIDIA.")
        return False
    if module_available("torch"):
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                return True
        except Exception:
            pass
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except (OSError, subprocess.SubprocessError):
            pass
    notes.append("CUDA image detected, but no NVIDIA device/toolkit is visible.")
    return False


def _vulkan_available(notes: list[str]) -> tuple[bool, str | None]:
    if sys.platform.startswith("linux"):
        render_nodes = list(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else []
        loader = any(
            Path(p).exists()
            for p in (
                "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
                "/usr/lib64/libvulkan.so.1",
                "/usr/lib/libvulkan.so.1",
            )
        )
        if not render_nodes:
            notes.append("No /dev/dri/renderD* node: Vulkan devices are not visible.")
            return False, None
        if not loader:
            notes.append("/dev/dri is present but the Vulkan loader is missing.")
            return False, None
        name = "Vulkan device"
        if shutil.which("vulkaninfo"):
            try:
                result = subprocess.run(
                    ["vulkaninfo", "--summary"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                for line in result.stdout.splitlines():
                    if "deviceName" in line:
                        name = line.split("=", 1)[-1].strip()
                        break
            except (OSError, subprocess.SubprocessError):
                pass
        return True, name
    if shutil.which("vulkaninfo"):
        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=2, check=False
            )
            if result.returncode == 0:
                name = "Vulkan device"
                for line in result.stdout.splitlines():
                    if "deviceName" in line:
                        name = line.split("=", 1)[-1].strip()
                        break
                return True, name
        except (OSError, subprocess.SubprocessError):
            pass
    notes.append("Vulkan is not available on this host (CPU-only dev environment).")
    return False, None


def probe_hardware(env: EnvConfig) -> HardwareStatus:
    notes: list[str] = []
    cores = os.cpu_count() or 1
    ram = _ram_gb()
    devices: dict[str, DeviceInfo] = {
        "cpu": DeviceInfo(platform=_cpu_name(), slug="cpu"),
    }
    if env.disable_hw_probe:
        notes.append("Hardware probe disabled by WS_DISABLE_HW_PROBE; assuming CPU only.")
        return HardwareStatus(
            image_variant=env.image_variant,
            devices=devices,
            cpu_cores=cores,
            ram_gb=ram,
            vulkan_ok=False,
            cuda_ok=False,
            notes=notes,
        )

    cuda_ok = _cuda_available(env.image_variant, notes)
    if cuda_ok:
        devices["cuda"] = DeviceInfo(platform="CUDA device", slug="cuda")
    vulkan_ok, vulkan_name = _vulkan_available(notes)
    if vulkan_ok:
        devices["vulkan"] = DeviceInfo(platform=vulkan_name or "Vulkan device", slug="vulkan")

    if not cuda_ok and not vulkan_ok:
        notes.append("CPU is the only available device; engines still queue and run, just slowly.")
    return HardwareStatus(
        image_variant=env.image_variant,
        devices=devices,
        cpu_cores=cores,
        ram_gb=ram,
        vulkan_ok=vulkan_ok,
        cuda_ok=cuda_ok,
        notes=notes,
    )
