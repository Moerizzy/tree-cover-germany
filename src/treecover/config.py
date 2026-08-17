"""Configuration loading — paths, state metadata and training settings.

Nothing in ``treecover`` hardcodes a filesystem path. Everything comes from
``configs/paths.yaml`` (machine-specific, gitignored) and
``configs/states.yaml`` (checked in, identical everywhere), so the same code
runs on the inference container, the HDD host and a reviewer's laptop.

Typical use::

    from treecover.config import load_paths, load_states

    paths = load_paths()
    states = load_states()
    by = states["BY"]
    print(by.lidar_url("706_5585"))
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

#: Repository root — three levels up from this file (src/treecover/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

_INTERPOLATION_RE = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")
_ENV_PREFIX = "TREECOVER_"


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════


class Paths(dict):
    """Path configuration with ``${key}`` interpolation and dotted lookup.

    Behaves like a nested dict but adds :meth:`get_path`, which resolves a
    dotted key and returns a :class:`~pathlib.Path`.
    """

    def get_path(self, dotted: str, default: str | None = None) -> Path:
        """Resolve a dotted key such as ``"sampling.clc"`` to a Path.

        Raises:
            KeyError: If the key is absent and no default is given.
        """
        value = self.get_value(dotted, default)
        if value is None:
            raise KeyError(
                f"'{dotted}' is not set in paths.yaml. Copy configs/paths.example.yaml "
                f"to configs/paths.yaml and fill it in, or set {_env_name(dotted)}."
            )
        return Path(str(value)).expanduser()

    def get_value(self, dotted: str, default: Any = None) -> Any:
        """Resolve a dotted key to its raw value."""
        env = os.environ.get(_env_name(dotted))
        if env is not None:
            return env
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _env_name(dotted: str) -> str:
    return _ENV_PREFIX + dotted.replace(".", "_").upper()


def _interpolate(raw: dict) -> dict:
    """Expand ``${dotted.key}`` references against the same document.

    Resolution is iterative so that ``a: ${b}`` and ``b: ${c}`` both work
    regardless of declaration order. Unresolvable references are left as-is
    rather than raising, so a partially filled config still loads.
    """

    def lookup(dotted: str) -> str | None:
        node: Any = raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node if isinstance(node, str) else None

    def walk(node: Any, depth: int = 0) -> Any:
        if isinstance(node, dict):
            return {k: walk(v, depth) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, depth) for v in node]
        if isinstance(node, str) and "${" in node and depth < 10:
            def sub(m: re.Match) -> str:
                resolved = lookup(m.group(1))
                return m.group(0) if resolved is None else resolved
            expanded = _INTERPOLATION_RE.sub(sub, node)
            return walk(expanded, depth + 1) if expanded != node else expanded
        return node

    return walk(raw)


@lru_cache(maxsize=None)
def load_paths(path: str | Path | None = None) -> Paths:
    """Load ``configs/paths.yaml`` (or an explicit file).

    Falls back to ``paths.example.yaml`` with a warning-free but clearly
    marked result, so ``--help`` and tests work in a fresh clone.

    Args:
        path: Explicit config file. Defaults to ``configs/paths.yaml``.

    Returns:
        A :class:`Paths` mapping with interpolation applied.
    """
    if path is not None:
        cfg_file = Path(path)
    else:
        cfg_file = CONFIG_DIR / "paths.yaml"
        if not cfg_file.exists():
            cfg_file = CONFIG_DIR / "paths.example.yaml"
    if not cfg_file.exists():
        return Paths()
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    return Paths(_interpolate(raw))


# ══════════════════════════════════════════════════════════════════════════════
# States
# ══════════════════════════════════════════════════════════════════════════════


def strip_zone_prefix(tile_name: str) -> str:
    """``'32706_5585'`` -> ``'706_5585'`` (Bavaria's download naming)."""
    easting, _, northing = tile_name.partition("_")
    if easting.startswith("32") and len(easting) > 3:
        easting = easting[2:]
    return f"{easting}_{northing}"


def zone_dash(tile_name: str) -> str:
    """``'33304_5862'`` -> ``'33304-5862'`` (Brandenburg's LAZ naming)."""
    return tile_name.replace("_", "-")


def keep(tile_name: str) -> str:
    """Identity — the tile name is already in download form."""
    return tile_name


#: Named transforms referenced by ``tile_name_rule`` in states.yaml.
TILE_NAME_RULES = {
    "strip_zone_prefix": strip_zone_prefix,
    "zone_dash": zone_dash,
    "keep": keep,
}


@dataclass(frozen=True)
class StateConfig:
    """Everything that varies between federal states."""

    code: str
    name: str
    gadm_hasc: str
    pred_dir: str
    epsg: int
    utm_zone: int
    tile_size_m: int
    target_resolution_m: float
    aliases: tuple[str, ...] = ()
    ortho: dict = field(default_factory=dict)
    lidar: dict = field(default_factory=dict)
    ndsm: dict = field(default_factory=dict)

    def _url(self, source: dict, tile_name: str) -> str | None:
        template = source.get("url_template")
        if not template:
            return None
        rule = TILE_NAME_RULES[source.get("tile_name_rule") or "keep"]
        return template.format(tile_name=rule(tile_name))

    def ortho_url(self, tile_name: str) -> str | None:
        """Download URL for an orthophoto tile, or None if not templated.

        States served via WCS/WMS/Atom feed return None; use the endpoint in
        ``self.ortho`` instead.
        """
        return self._url(self.ortho, tile_name)

    def lidar_url(self, tile_name: str) -> str | None:
        """Download URL for a LiDAR tile, or None if the state has no LAZ feed."""
        return self._url(self.lidar, tile_name)

    @property
    def has_lidar(self) -> bool:
        """Whether per-tile LiDAR downloads are configured for this state."""
        return bool(self.lidar.get("url_template"))


class StateRegistry(dict):
    """Lookup of :class:`StateConfig` by code, tolerating known aliases."""

    def __getitem__(self, key: str) -> StateConfig:
        code = key.upper()
        if code in self.keys():
            return super().__getitem__(code)
        for state in self.values():
            if code in state.aliases:
                return state
        raise KeyError(
            f"Unknown state '{key}'. Known: {', '.join(sorted(self.keys()))}."
        )

    def resolve(self, key: str) -> str:
        """Return the canonical code for ``key`` (``'NRW'`` -> ``'NW'``)."""
        return self[key].code


@lru_cache(maxsize=None)
def load_states(path: str | Path | None = None) -> StateRegistry:
    """Load ``configs/states.yaml`` into a :class:`StateRegistry`."""
    cfg_file = Path(path) if path else CONFIG_DIR / "states.yaml"
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults", {})
    registry = StateRegistry()
    for code, entry in (raw.get("states") or {}).items():
        registry[code] = StateConfig(
            code=code,
            name=entry["name"],
            gadm_hasc=entry["gadm_hasc"],
            pred_dir=entry.get("pred_dir", code),
            epsg=entry.get("epsg", defaults.get("epsg", 25832)),
            utm_zone=entry.get("utm_zone", defaults.get("utm_zone", 32)),
            tile_size_m=entry.get("tile_size_m", defaults.get("tile_size_m", 1000)),
            target_resolution_m=entry.get(
                "target_resolution_m", defaults.get("target_resolution_m", 0.20)
            ),
            aliases=tuple(entry.get("aliases", ())),
            ortho=entry.get("ortho") or {},
            lidar=entry.get("lidar") or {},
            ndsm=entry.get("ndsm") or {},
        )
    return registry


@lru_cache(maxsize=None)
def validation_states(path: str | Path | None = None) -> tuple[str, ...]:
    """State codes with LiDAR reference data (the accuracy-assessment set)."""
    cfg_file = Path(path) if path else CONFIG_DIR / "states.yaml"
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    return tuple(raw.get("validation_states", ()))
