"""Pytest fixtures and workspace isolation guards for SkyCast E2E tests."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from skycast.config.settings import Settings, get_settings
from skycast.data.storage import SkyCastStorage
from tests.e2e.fakes import ContractSupabaseFake, MockHopsworksProject
from tests.e2e.fixtures import (
    generate_lahore_feature_dataframe,
    make_e2e_settings,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_LOCAL_DIR = _REPO_ROOT / ".skycast_local"
_REPO_LOCAL_DB = _REPO_LOCAL_DIR / "skycast.db"

_ENV_PREFIXES: tuple[str, ...] = (
    "SKYCAST_",
    "SUPABASE_",
    "HOPSWORKS_",
    "GROQ_",
    "GEMINI_",
    "GITHUB_",
    "PRIMARY_LLM_",
    "FALLBACK_LLM_",
    "TERTIARY_LLM_",
)


def _patch_settings_everywhere(monkeypatch: pytest.MonkeyPatch, cfg: Settings) -> None:
    def _get_cfg() -> Settings:
        return cfg

    monkeypatch.setenv("SKYCAST_ENV", cfg.skycast_env)
    monkeypatch.setenv("SKYCAST_OFFLINE_MODE", cfg.skycast_offline_mode)
    monkeypatch.setenv("SKYCAST_LOCAL_DATA_DIR", str(cfg.skycast_local_data_dir))
    monkeypatch.setenv("FORECAST_HORIZON_HOURS", str(cfg.forecast_horizon_hours))

    import skycast.config as skycast_config

    settings_mod = sys.modules["skycast.config.settings"]
    monkeypatch.setattr(settings_mod, "get_settings", _get_cfg)
    monkeypatch.setattr(settings_mod, "settings", cfg)
    monkeypatch.setattr(skycast_config, "get_settings", _get_cfg)
    monkeypatch.setattr(skycast_config, "settings", cfg)

    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("skycast.") and mod is not None and hasattr(mod, "get_settings"):
            monkeypatch.setattr(mod, "get_settings", _get_cfg)


@pytest.fixture(autouse=True)
def _isolate_e2e_env_and_guard_local_db(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Clear cached settings/env vars and block any in-process SQLite access to `d:\\SkyCast\\.skycast_local`."""
    get_settings.cache_clear()

    for key in list(os.environ.keys()):
        if key.upper().startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)

    orig_sqlite_connect = sqlite3.connect
    repo_local_resolved = _REPO_LOCAL_DIR.resolve()

    def _guarded_sqlite_connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if isinstance(database, (str, Path)) and str(database) != ":memory:":
            db_path = Path(database).resolve()
            if db_path == repo_local_resolved or repo_local_resolved in db_path.parents:
                raise AssertionError(
                    f"E2E isolation violation: attempted sqlite3.connect to repo path {db_path}"
                )
        return orig_sqlite_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _guarded_sqlite_connect)

    yield

    get_settings.cache_clear()


@pytest.fixture()
def e2e_tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Offline `Settings` rooted in `tmp_path/.skycast_local` with `get_settings()` patched."""
    cfg = make_e2e_settings(tmp_path, offline=True)
    _patch_settings_everywhere(monkeypatch, cfg)
    return cfg


@pytest.fixture()
def offline_e2e_storage(e2e_tmp_settings: Settings) -> SkyCastStorage:
    """Initialized `SkyCastStorage` in `local_offline` mode inside `tmp_path`."""
    storage = SkyCastStorage(settings=e2e_tmp_settings)
    storage.init_local_db()
    return storage


@pytest.fixture()
def contract_supabase_fake() -> ContractSupabaseFake:
    """Fresh strict `ContractSupabaseFake` instance."""
    return ContractSupabaseFake(strict_domain_enums=True)


@pytest.fixture()
def mock_hopsworks_project(tmp_path: Path) -> MockHopsworksProject:
    """Fresh `MockHopsworksProject` backed by `tmp_path/hopsworks_artifacts`."""
    return MockHopsworksProject(artifact_root=tmp_path / "hopsworks_artifacts")


@pytest.fixture()
def dual_e2e_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_supabase_fake: ContractSupabaseFake,
    mock_hopsworks_project: MockHopsworksProject,
) -> SkyCastStorage:
    """`cloud_dual` `SkyCastStorage` wired to `ContractSupabaseFake` + `MockHopsworksProject`."""
    cfg = make_e2e_settings(tmp_path, offline=False)
    _patch_settings_everywhere(monkeypatch, cfg)

    storage = SkyCastStorage(settings=cfg)
    storage.init_local_db()
    storage._supabase_client = contract_supabase_fake
    storage._hopsworks_project = mock_hopsworks_project
    monkeypatch.setattr(storage, "_get_supabase", lambda: contract_supabase_fake)
    monkeypatch.setattr(
        storage, "_get_hopsworks_feature_store", lambda: mock_hopsworks_project.get_feature_store()
    )
    try:
        import hopsworks as _hopsworks_mod

        monkeypatch.setattr(_hopsworks_mod, "login", mock_hopsworks_project.login)
    except ImportError:
        pass

    storage.supabase_fake = contract_supabase_fake  # type: ignore[attr-defined]
    storage.hopsworks_project = mock_hopsworks_project  # type: ignore[attr-defined]
    return storage


@pytest.fixture()
def lahore_30d_df() -> pd.DataFrame:
    """Deterministic 30-day (720h) Lahore engineered+targeted DataFrame (H=24)."""
    return generate_lahore_feature_dataframe(n_hours=720, horizon_hours=24, seed=42)


@pytest.fixture()
def lahore_custom_h12_df() -> pd.DataFrame:
    """Deterministic 30-day (720h) Lahore engineered+targeted DataFrame (H=12)."""
    return generate_lahore_feature_dataframe(n_hours=720, horizon_hours=12, seed=42)
