"""Regression tests for the LanceDB backend restoration (issue #34).

The LanceDB backend module (``mnema.backends.lancedb``) still ships in the
source tree, but the wiring around it was dropped: the ``lancedb``
packaging extra, the ``BackendName`` literal entry, and the ``make_backend``
factory branch. These tests pin the restored behavior:

* packaging metadata — the ``lancedb`` extra exists with the exact pinned
  spec and is included in ``all`` (plus keyword / mypy / pytest-marker
  metadata), and
* runtime wiring — a config declaring ``backend="lancedb"`` validates and
  reaches the LanceDB factory branch, offline (a fake
  ``mnema.backends.lancedb`` module is injected, so no real ``lancedb``
  install, network, or credentials are needed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import get_args

import pytest
import tomllib

from mnema.backends import make_backend
from mnema.config import BackendName, MnemaConfig
from mnema.errors import ConfigError

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _lancedb_config(tmp_path: Path) -> MnemaConfig:
    """Build a config declaring ``backend="lancedb"``.

    A rejected literal is a spec violation, so surface it as an assertion
    failure rather than a raw pydantic error.
    """
    try:
        return MnemaConfig(backend="lancedb", backend_path=str(tmp_path / "lancedb"))
    except Exception as exc:
        raise AssertionError(f"MnemaConfig rejected backend='lancedb': {exc}") from exc


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


class TestLanceDBPackaging:
    """The ``lancedb`` optional-dependency group must be declared."""

    def test_lancedb_extra_exists(self, pyproject: dict) -> None:
        extras = pyproject["project"]["optional-dependencies"]
        assert "lancedb" in extras, (
            f"'lancedb' extra missing from [project.optional-dependencies]; "
            f"declared extras: {sorted(extras)}"
        )
        assert extras["lancedb"] == ["lancedb>=0.10"]

    def test_lancedb_extra_in_all(self, pyproject: dict) -> None:
        extras = pyproject["project"]["optional-dependencies"]
        assert "lancedb>=0.10" in extras["all"], "'all' extra must include the lancedb dependency"

    def test_lancedb_keyword_restored(self, pyproject: dict) -> None:
        assert "lancedb" in pyproject["project"]["keywords"]

    def test_lancedb_mypy_override(self, pyproject: dict) -> None:
        overrides = pyproject["tool"]["mypy"]["overrides"]
        assert any("lancedb.*" in override.get("module", []) for override in overrides), (
            "mypy override module list must include 'lancedb.*'"
        )

    def test_lancedb_pytest_marker(self, pyproject: dict) -> None:
        markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
        assert any(m.startswith("lancedb:") for m in markers), (
            "pytest markers must declare the 'lancedb' marker"
        )


class TestLanceDBBackendName:
    def test_lancedb_in_backend_name_literal(self) -> None:
        assert "lancedb" in get_args(BackendName), (
            f"BackendName must include 'lancedb'; got {get_args(BackendName)}"
        )

    def test_config_accepts_lancedb(self, tmp_path: Path) -> None:
        cfg = _lancedb_config(tmp_path)
        assert cfg.backend == "lancedb"


class FakeLanceDBBackend:
    """Offline stand-in for the real LanceDBBackend constructor."""

    name = "lancedb"

    def __init__(self, config: MnemaConfig) -> None:
        self.config = config


@pytest.fixture
def fake_lancedb_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Inject a fake ``mnema.backends.lancedb`` so make_backend never
    touches the real optional dependency (no install, no network)."""
    module = ModuleType("mnema.backends.lancedb")
    module.LanceDBBackend = FakeLanceDBBackend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mnema.backends.lancedb", module)
    return module


class TestLanceDBFactory:
    """``make_backend`` must dispatch ``backend="lancedb"`` to LanceDBBackend."""

    def test_lancedb_reaches_factory(self, fake_lancedb_module: ModuleType, tmp_path: Path) -> None:
        cfg = _lancedb_config(tmp_path)
        try:
            backend = make_backend(cfg)
        except ConfigError as exc:
            raise AssertionError(
                f"make_backend failed to dispatch backend='lancedb': {exc}"
            ) from exc
        assert isinstance(backend, FakeLanceDBBackend)
        assert backend.name == "lancedb"
        assert backend.config is cfg

    def test_lancedb_missing_dep_raises_not_available(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Without the optional dep, the factory must raise
        BackendNotAvailableError — not ConfigError ('unknown backend')."""
        import builtins

        real_import = builtins.__import__

        def blocked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "mnema.backends.lancedb":
                raise ImportError("No module named 'lancedb'")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "mnema.backends.lancedb", raising=False)
        monkeypatch.setattr(builtins, "__import__", blocked_import)

        from mnema.errors import BackendNotAvailableError

        cfg = _lancedb_config(tmp_path)
        with pytest.raises(BackendNotAvailableError):
            make_backend(cfg)
