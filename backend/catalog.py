"""Catalog loading and capability helpers.

The YAML in ``backend/db/catalog.yaml`` is the runtime mirror of docs/02 §2.
Nothing outside engine adapters may hard-code model-specific flags; UI gating
and validation flow from the helpers in this module.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

CATALOG_PATH = Path(__file__).with_name("db") / "catalog.yaml"

DeviceSlug = Literal["cpu", "cuda", "vulkan"]
WordTs = Literal["native", "aligner", "none"]
Role = Literal["asr", "diar", "aligner"]


class FetchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["single", "snapshot"]
    filename: str | None = None
    allow_patterns: list[str] = Field(default_factory=list)


class ProbeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modules: list[str] = Field(default_factory=list)
    executable: str | None = None
    executable_env: str | None = None
    builtin: bool = False


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    role: Role = "asr"
    repo: str | None = None
    fetch: FetchSpec | None = None
    size_bytes: int = 0
    size_label: str = "—"
    languages: str = ""
    language_codes: list[str] = Field(default_factory=list)
    language_auto: bool = True
    english_only: bool = False
    word_ts: WordTs = "none"
    max_speakers: int | None = None
    gated: bool = False
    notes: str = ""
    default: bool = False
    engine: str = ""

    @property
    def downloadable(self) -> bool:
        return bool(self.repo and self.fetch)


class EngineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    runtime: str
    kind: Literal["asr", "diar", "aligner", "demo"]
    devices: list[DeviceSlug]
    probe: ProbeSpec
    license: str = ""
    notes: str = ""
    hidden: bool = False
    models: list[ModelSpec] = Field(default_factory=list)

    def models_for(self, role: Role) -> list[ModelSpec]:
        return [m for m in self.models if m.role == role]

    def supports_device(self, slug: str) -> bool:
        return slug in self.devices


class LanguageTable(BaseModel):
    whisper: list[str] = Field(default_factory=list)
    nemo: list[str] = Field(default_factory=list)
    qwen3: list[str] = Field(default_factory=list)
    aligner: list[str] = Field(default_factory=list)


class Catalog(BaseModel):
    version: int
    schema_version: int
    engines: list[EngineSpec]
    languages: LanguageTable

    def engine(self, key: str) -> EngineSpec:
        for engine in self.engines:
            if engine.key == key:
                return engine
        raise KeyError(key)

    def engine_or_none(self, key: str) -> EngineSpec | None:
        return next((e for e in self.engines if e.key == key), None)

    def model(self, model_id: str) -> ModelSpec:
        for engine in self.engines:
            for model in engine.models:
                if model.id == model_id:
                    return model
        raise KeyError(model_id)

    def model_or_none(self, model_id: str) -> ModelSpec | None:
        for engine in self.engines:
            for model in engine.models:
                if model.id == model_id:
                    return model
        return None

    @property
    def visible_engines(self) -> list[EngineSpec]:
        return [e for e in self.engines if not e.hidden]

    def asr_models(self) -> list[ModelSpec]:
        return [m for e in self.visible_engines for m in e.models if m.role == "asr"]

    def diar_models(self) -> list[ModelSpec]:
        return [m for e in self.visible_engines for m in e.models if m.role == "diar"]

    def aligner_models(self) -> list[ModelSpec]:
        return [m for e in self.visible_engines for m in e.models if m.role == "aligner"]

    def diar_models_for(self, family: str) -> list[ModelSpec]:
        engine_key = {"nemo": "nemo-speech", "pyannote": "pyannote"}.get(family)
        if engine_key is None:
            return []
        engine = self.engine_or_none(engine_key)
        return engine.models_for("diar") if engine else []

    def default_asr_model(self) -> ModelSpec:
        for model in self.asr_models():
            if model.default:
                return model
        return self.asr_models()[0]

    def language_options(self, model: ModelSpec) -> list[str]:
        if model.language_codes:
            return model.language_codes
        engine = self.engine_or_none(model.engine)
        if engine is None:
            return []
        if engine.key in {"faster-whisper", "whisper.cpp", "demo"}:
            return self.languages.whisper
        if engine.key == "nemo-speech":
            return self.languages.nemo
        if engine.key == "qwen3-asr":
            return self.languages.qwen3
        if engine.key == "qwen3-aligner":
            return self.languages.aligner
        return []

    def aligner_model(self) -> ModelSpec | None:
        models = self.aligner_models()
        return models[0] if models else None

    def is_whisper_family(self, model: ModelSpec) -> bool:
        return model.engine in {"faster-whisper", "whisper.cpp"}


def _fill_defaults(catalog: Catalog) -> None:
    for engine in catalog.engines:
        engine_key = engine.key
        for model in engine.models:
            model.engine = engine_key
            if not model.language_codes:
                if engine_key in {"faster-whisper", "whisper.cpp"}:
                    model.language_codes = list(catalog.languages.whisper)
                elif engine_key == "nemo-speech":
                    model.language_codes = list(catalog.languages.nemo)
                elif engine_key == "qwen3-asr":
                    model.language_codes = list(catalog.languages.qwen3)
                elif engine_key == "qwen3-aligner":
                    model.language_codes = list(catalog.languages.aligner)


def load_catalog(path: Path | str | None = None) -> Catalog:
    path = Path(path or os.environ.get("WS_CATALOG_PATH") or CATALOG_PATH)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    catalog = Catalog.model_validate(raw)
    _fill_defaults(catalog)

    seen: set[str] = set()
    for engine in catalog.engines:
        if not engine.models and engine.key != "qwen3-aligner":
            raise ValueError(f"engine {engine.key!r} has no models")
        for model in engine.models:
            if model.id in seen:
                raise ValueError(f"duplicate model id {model.id!r}")
            seen.add(model.id)
            if model.role == "asr" and engine.kind not in {"asr", "demo"}:
                raise ValueError(f"ASR model {model.id!r} belongs to non-ASR engine {engine.key!r}")
            if model.gated and not model.repo:
                raise ValueError(f"gated model {model.id!r} needs a repo")
    return catalog


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def engine_available(engine: EngineSpec) -> bool:
    """True when the runtime for this engine is present in this environment."""
    probe = engine.probe
    if probe.builtin:
        return True
    if probe.executable:
        env_path = os.environ.get(probe.executable_env, "") if probe.executable_env else ""
        if env_path and Path(env_path).exists():
            return True
        return bool(shutil.which(probe.executable))
    return all(module_available(module) for module in probe.modules)


def capability_matrix(catalog: Catalog) -> dict[str, dict[str, str | bool]]:
    """Compact machine-readable copy of docs/02 §5, used by tests and the UI."""
    result: dict[str, dict[str, str | bool]] = {}
    for engine in catalog.visible_engines:
        for model in engine.models:
            if model.role != "asr":
                continue
            result[model.id] = {
                "engine": engine.key,
                "language_auto": model.language_auto,
                "word_timestamps": model.word_ts,
                "captions": True,
                "interval_timestamps": True,
                "turn_timestamps": True,
                "pauses": model.word_ts != "none",
                "verbatim": engine.key in {"faster-whisper", "whisper.cpp"},
            }
    return result
