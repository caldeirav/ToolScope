"""Tests for eval/local/models.yaml registry."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MODELS_YAML = Path(__file__).resolve().parents[1] / "eval" / "local" / "models.yaml"
LOCAL_CONFIG = Path(__file__).resolve().parents[1] / "eval" / "local" / "bfcl_multiple_local.yaml"

EXPECTED_MODELS = {
    "llama-3.2-3b-instruct",
    "qwen2.5-7b-instruct",
    "llama-3.1-8b-instruct",
    "qwen3-32b",
    "llama-3.3-70b-instruct",
}


@pytest.fixture
def models_registry() -> dict:
    return yaml.safe_load(MODELS_YAML.read_text()) or {}


def test_models_yaml_has_five_entries(models_registry: dict):
    models = models_registry.get("models") or {}
    assert len(models) == 5
    assert set(models) == EXPECTED_MODELS


def test_slm_tier_entries(models_registry: dict):
    models = models_registry.get("models") or {}
    for mid in ("llama-3.2-3b-instruct", "qwen2.5-7b-instruct"):
        assert mid in models
        assert models[mid]["file_glob"] == "*Q8_0*.gguf"


def test_tier_metadata(models_registry: dict):
    models = models_registry.get("models") or {}
    tiers = models_registry.get("tiers") or {}
    assert "high_sensitivity" in tiers
    assert "mid_production" in tiers
    assert "control_ceiling" in tiers
    assert models["qwen2.5-7b-instruct"]["tier"] == "high_sensitivity"
    assert models["llama-3.1-8b-instruct"]["tier"] == "mid_production"
    assert models["qwen3-32b"]["tier"] == "mid_production"
    assert models["llama-3.3-70b-instruct"]["tier"] == "control_ceiling"


def test_mid_tier_single_file_quants(models_registry: dict):
    models = models_registry.get("models") or {}
    for mid in ("llama-3.1-8b-instruct", "qwen3-32b"):
        spec = models[mid]
        assert "hf_include" in spec
        assert "Q4_K_M" in spec["file_glob"]
        assert spec["context_size"] == 32768


def test_model_aliases_unique(models_registry: dict):
    models = models_registry.get("models") or {}
    aliases = [m["alias"] for m in models.values()]
    assert len(aliases) == len(set(aliases))


def test_required_model_fields(models_registry: dict):
    required = {"hf_repo", "file_glob", "alias"}
    for model_id, spec in (models_registry.get("models") or {}).items():
        missing = required - set(spec)
        assert not missing, f"{model_id} missing {missing}"


def test_local_bfcl_config_matches_registry(models_registry: dict):
    cfg = yaml.safe_load(LOCAL_CONFIG.read_text()) or {}
    registry_aliases = {m["alias"] for m in (models_registry.get("models") or {}).values()}
    entry_names = {e["name"] for e in cfg["model"]["entries"]}
    assert entry_names == registry_aliases

    assert cfg["dataset"]["protocol"] == "shared_catalog"
    assert cfg["dataset"]["pool_size"] is None
    assert cfg["toolscope"]["k_values"] == [5, 10, 20]
    assert cfg["output"]["versioned_dir"] == "eval/paper/artifacts/local"
