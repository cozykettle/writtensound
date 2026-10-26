from __future__ import annotations

from backend.catalog import load_catalog

# Roster replicated from docs/02 §2 so a catalog edit cannot drift silently.
EXPECTED_MODELS = {
    "faster-whisper": {
        "fw-large-v3",
        "fw-large-v3-turbo",
        "fw-distil-large-v3.5",
        "fw-small",
    },
    "whisper.cpp": {
        "wcpp-large-v3-turbo-f16",
        "wcpp-large-v3-turbo-q8_0",
        "wcpp-large-v3-turbo-q5_0",
        "wcpp-large-v3-q5_0",
        "wcpp-distil-large-v3",
    },
    "nemo-speech": {
        "nemo-nemotron-3.5-asr",
        "nemo-nemotron-en",
        "nemo-parakeet-tdt-v3",
        "nemo-diar-nemotron-3",
        "nemo-diar-sortformer-4spk-v2.1",
    },
    "qwen3-asr": {"qwen3-asr-0.6b", "qwen3-asr-1.7b"},
    "qwen3-aligner": {"qwen3-aligner-0.6b"},
    "pyannote": {"pyannote-community-1", "pyannote-3.1"},
    "demo": {"demo-script"},
}


def test_catalog_roster_matches_docs() -> None:
    catalog = load_catalog()
    assert {engine.key for engine in catalog.engines} == set(EXPECTED_MODELS)
    for engine in catalog.engines:
        assert {model.id for model in engine.models} == EXPECTED_MODELS[engine.key]
        for model in engine.models:
            assert model.engine == engine.key
            assert model.role in {"asr", "diar", "aligner"}


def test_default_and_capability_helpers_are_consistent() -> None:
    catalog = load_catalog()
    default = catalog.default_asr_model()
    assert default.id == "wcpp-large-v3-turbo-q8_0"
    assert default.role == "asr"

    aligner = catalog.aligner_model()
    assert aligner is not None
    assert aligner.id == "qwen3-aligner-0.6b"

    assert catalog.model("demo-script").word_ts == "native"
    assert catalog.model("qwen3-asr-0.6b").word_ts == "aligner"
    assert catalog.is_whisper_family(catalog.model("fw-large-v3"))
    assert catalog.is_whisper_family(catalog.model("wcpp-large-v3-q5_0"))
    assert not catalog.is_whisper_family(catalog.model("nemo-nemotron-3.5-asr"))

    assert {"pyannote-community-1", "pyannote-3.1"} == {
        model.id for model in catalog.diar_models_for("pyannote")
    }
    assert {"nemo-diar-nemotron-3", "nemo-diar-sortformer-4spk-v2.1"} == {
        model.id for model in catalog.diar_models_for("nemo")
    }


def test_qwen_snapshots_fetch_separate_chat_template() -> None:
    catalog = load_catalog()
    for model_id in ("qwen3-asr-0.6b", "qwen3-asr-1.7b", "qwen3-aligner-0.6b"):
        fetch = catalog.model(model_id).fetch
        assert fetch is not None
        assert "*.jinja" in fetch.allow_patterns


def test_english_only_and_max_speaker_metadata() -> None:
    catalog = load_catalog()
    assert catalog.model("fw-distil-large-v3.5").english_only
    assert catalog.model("wcpp-distil-large-v3").english_only
    assert catalog.model("nemo-diar-sortformer-4spk-v2.1").max_speakers == 4
