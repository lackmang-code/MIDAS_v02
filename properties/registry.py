# -*- coding: utf-8 -*-
"""
properties/registry.py
──────────────────────
물성 디렉토리를 스캔하여 PropertyConfig 객체로 로드.

규약
  properties/<name>/config.yaml    : 메타데이터
  properties/<name>/dataset.csv    : 분자 데이터 (smiles + target)
  properties/<name>/paper_template.py (선택)  : 논문 도메인 텍스트

YAML이 아닌 단순 KEY: VALUE 파서를 사용하여 PyYAML 의존성 회피.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import PropertyConfig

# ── YAML 의존성 없는 미니 파서 ────────────────────────────────────────────────
def _strip_inline_comment(s: str) -> str:
    """리스트/문자열 안의 # 은 보존, 그 외 위치의 # 이후는 주석으로 제거."""
    in_str: str | None = None
    depth = 0
    for i, ch in enumerate(s):
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "#" and depth == 0:
            return s[:i]
    return s


def _parse_value(raw: str) -> Any:
    s = _strip_inline_comment(raw).strip()
    if not s:
        return ""
    # 리스트:  [a, b, c]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        items = [_parse_value(x) for x in inner.split(",")]
        return items
    # 따옴표 문자열
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    # bool
    if s.lower() in ("true", "yes"):  return True
    if s.lower() in ("false", "no"):  return False
    if s.lower() in ("null", "none", "~"): return None
    # 숫자
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def _load_mini_yaml(path: Path) -> dict:
    out: dict = {}
    cur_key: str | None = None
    cur_dict: dict | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            # 들여쓰기된 라인 = extra 딕셔너리 항목
            if line.startswith((" ", "\t")) and cur_dict is not None:
                k, _, v = line.strip().partition(":")
                cur_dict[k.strip()] = _parse_value(v)
                continue
            # top-level key: value
            cur_dict = None
            k, _, v = line.partition(":")
            key = k.strip()
            val = v.strip()
            if not val:
                # 다음 줄들이 들여쓰기된 dict
                out[key] = {}
                cur_dict = out[key]
                cur_key = key
            else:
                out[key] = _parse_value(val)
    return out


# ── 레지스트리 본체 ───────────────────────────────────────────────────────────
_PROPERTIES_DIR = Path(__file__).parent


def _build_config(prop_dir: Path) -> PropertyConfig:
    meta = _load_mini_yaml(prop_dir / "config.yaml")
    dataset_path = prop_dir / meta.get("dataset_file", "dataset.csv")

    # tuple로 변환 (frozen dataclass)
    stratify = tuple(float(x) for x in meta.get("stratify_bins", []))
    emphasis = tuple(meta.get("descriptor_emphasis", []) or [])

    return PropertyConfig(
        name                = meta["name"],
        display_name        = meta.get("display_name", meta["name"]),
        domain              = meta.get("domain", ""),
        dataset_path        = dataset_path,
        smiles_column       = meta.get("smiles_column", "smiles"),
        target_column       = meta["target_column"],
        unit                = meta.get("unit", ""),
        task_type           = meta.get("task_type", "regression"),
        log_transform       = bool(meta.get("log_transform", True)),
        stratify_bins       = stratify,
        direction           = meta.get("direction", "lower_better"),
        screening_threshold = float(meta.get("screening_threshold", 0.0)),
        training_threshold  = (None if meta.get("training_threshold") is None
                               else float(meta["training_threshold"])),
        descriptor_emphasis = emphasis,
        description_short   = meta.get("description_short", ""),
        extra               = meta.get("extra", {}) or {},
    )


def list_properties() -> list[str]:
    """등록된 물성 이름 목록 (config.yaml 존재하는 디렉토리만)."""
    names = []
    for p in sorted(_PROPERTIES_DIR.iterdir()):
        if p.is_dir() and (p / "config.yaml").exists():
            names.append(p.name)
    return names


def load_property(name: str) -> PropertyConfig:
    """이름으로 PropertyConfig 로드."""
    prop_dir = _PROPERTIES_DIR / name
    if not (prop_dir / "config.yaml").exists():
        raise ValueError(
            f"Property '{name}' not found. "
            f"Available: {list_properties()}"
        )
    return _build_config(prop_dir)


def load_all() -> dict[str, PropertyConfig]:
    """등록된 모든 물성 로드 (UI 드롭다운용)."""
    return {n: load_property(n) for n in list_properties()}
