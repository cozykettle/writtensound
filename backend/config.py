"""Environment contract, app-config.json persistence, and filesystem layout."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .catalog import Catalog
from .jobs import JobSettings, OutputFormats

HF_TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{30,}$")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def default_settings() -> JobSettings:
    return JobSettings(
        asr_model_id="wcpp-large-v3-turbo-q8_0",
        output=OutputFormats(html=True, txt=True, srt=True, vtt=True, json=True),
    )


class DefaultPreselect(BaseModel):
    asr_model_id: str = "wcpp-large-v3-turbo-q8_0"
    device: str = "auto"
    language: str = "auto"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    hf_token: str | None = None
    defaults: JobSettings = Field(default_factory=default_settings)
    engine_enabled: dict[str, bool] = Field(default_factory=dict)
    keep_completed_jobs_days: int = 14
    cleanup_uploads: bool = True
    default_preselect: DefaultPreselect = Field(default_factory=DefaultPreselect)

    @field_validator("hf_token")
    @classmethod
    def _token(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not HF_TOKEN_RE.match(value):
            raise ValueError("hf_token must look like hf_<at least 30 alphanumerics>")
        return value

    @field_validator("keep_completed_jobs_days")
    @classmethod
    def _retention(cls, value: int) -> int:
        return max(0, min(3650, value))


class SettingsUpdate(BaseModel):
    """Subset of editable app-config keys. Unset keys are preserved."""

    model_config = ConfigDict(extra="forbid")

    hf_token: str | None = None
    defaults: JobSettings | None = None
    engine_enabled: dict[str, bool] | None = None
    keep_completed_jobs_days: int | None = None
    cleanup_uploads: bool | None = None
    default_preselect: DefaultPreselect | None = None

    @field_validator("hf_token")
    @classmethod
    def _token(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not HF_TOKEN_RE.match(value):
            raise ValueError("hf_token must look like hf_<at least 30 alphanumerics>")
        return value


@dataclass(frozen=True)
class EnvConfig:
    data_dir: Path
    port: int
    app_password: str | None
    hf_token_env: str | None
    hf_home: Path
    log_level: str
    disable_hw_probe: bool
    image_variant: str
    max_upload_bytes: int
    demo_engine_enabled: bool

    @classmethod
    def from_env(cls) -> EnvConfig:
        data_dir = Path(os.environ.get("WS_DATA_DIR", "./data")).expanduser().resolve()
        hf_home = Path(os.environ.get("HF_HOME", str(data_dir / "models" / "hf-cache"))).expanduser().resolve()
        variant = os.environ.get("WS_IMAGE_VARIANT", "standard").strip().lower()
        if variant not in {"standard", "cuda"}:
            variant = "standard"
        try:
            port = int(os.environ.get("WS_PORT", "8080"))
        except ValueError:
            port = 8080
        try:
            max_upload = int(os.environ.get("WS_MAX_UPLOAD_BYTES", str(16 * 1024**3)))
        except ValueError:
            max_upload = 16 * 1024**3
        return cls(
            data_dir=data_dir,
            port=port,
            app_password=os.environ.get("APP_PASSWORD") or None,
            hf_token_env=os.environ.get("HF_TOKEN") or None,
            hf_home=hf_home,
            log_level=os.environ.get("WS_LOG_LEVEL", "INFO").upper(),
            disable_hw_probe=env_bool("WS_DISABLE_HW_PROBE"),
            image_variant=variant,
            max_upload_bytes=max_upload,
            demo_engine_enabled=env_bool("WS_ENABLE_DEMO_ENGINE", True),
        )


@dataclass(frozen=True)
class Paths:
    root: Path
    config_dir: Path
    jobs: Path
    staging: Path
    models: Path
    hf_cache: Path

    @classmethod
    def build(cls, data_dir: Path, hf_home: Path) -> Paths:
        root = data_dir
        return cls(
            root=root,
            config_dir=root / "config",
            jobs=root / "jobs",
            staging=root / "jobs" / "staging",
            models=root / "models",
            hf_cache=hf_home,
        )

    @property
    def config_file(self) -> Path:
        return self.config_dir / "app-config.json"

    @property
    def db_file(self) -> Path:
        return self.jobs / "writtensound.db"

    def engine_models(self, engine_key: str, model_id: str) -> Path:
        return self.models / engine_key.replace(".", "") / model_id

    def job_dir(self, job_id: str) -> Path:
        return self.jobs / job_id

    def create_dirs(self) -> None:
        for path in (
            self.config_dir,
            self.jobs,
            self.staging,
            self.models,
            self.hf_cache,
            self.models / "whispercpp",
            self.models / "nemo",
            self.models / "fastwhisper",
            self.models / "qwen3",
            self.models / "pyannote",
        ):
            path.mkdir(parents=True, exist_ok=True)


def default_engine_flags(catalog: Catalog) -> dict[str, bool]:
    return {engine.key: True for engine in catalog.engines}


def load_config(paths: Paths, catalog: Catalog) -> AppConfig:
    base = AppConfig(engine_enabled=default_engine_flags(catalog))
    if not paths.config_file.exists():
        return base
    try:
        raw: dict[str, Any] = json.loads(paths.config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return base
    merged = base.model_dump(mode="json")
    for key, value in raw.items():
        if key in merged and value is not None:
            merged[key] = value
    enabled = default_engine_flags(catalog)
    enabled.update({str(k): bool(v) for k, v in raw.get("engine_enabled", {}).items()})
    merged["engine_enabled"] = enabled
    return AppConfig.model_validate(merged)


def save_config(paths: Paths, config: AppConfig) -> None:
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    tmp = paths.config_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(paths.config_file)


def mask_token(token: str | None) -> str | None:
    if not token:
        return None
    return f"hf_****{token[-4:]}"


def effective_hf_token(config: AppConfig, env: EnvConfig) -> str | None:
    return config.hf_token or env.hf_token_env
