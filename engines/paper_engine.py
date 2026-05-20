# -*- coding: utf-8 -*-
"""
engines/paper_engine.py  (MIDAS_v02)
────────────────────────────────────
물성별 논문 초안 자동 생성 라우터.

라우팅 규칙
  - cfg.name == "dielectric"  → v01 paper template (_paper_dielectric.py)
  - 그 외 모든 물성             → _paper_generic.py 의 범용 템플릿

각 템플릿 모듈은 동일한 시그니처를 노출해야 한다:
    generate_paper(...) -> dict
    paper_to_docx(paper, figures=None) -> bytes
    generate_paper_figures(...) -> dict[str, bytes]
"""
from __future__ import annotations

from typing import Any

# 라우팅용 import는 lazy — 큰 v01 템플릿을 매번 로드하지 않음
def _route(cfg_name: str):
    if cfg_name == "dielectric":
        from . import _paper_dielectric as mod
    else:
        from . import _paper_generic as mod
    return mod


# ─────────────────────────────────────────────────────────────────────────────
def generate_paper(cfg, **kwargs) -> dict:
    """
    Parameters
    ----------
    cfg : PropertyConfig  — 첫 번째 위치 인자로 받는다 (라우팅 키)
    **kwargs : 각 템플릿이 요구하는 인자들 (df_dataset, metrics, cv_result, ...)
    """
    mod = _route(cfg.name)

    if cfg.name == "dielectric":
        # v01 시그니처 호환: cfg를 풀어서 전달
        kwargs.setdefault("k_threshold",     cfg.screening_threshold)
        kwargs.setdefault("train_threshold", cfg.training_threshold or 3.5)
        return mod.generate_paper(**kwargs)

    return mod.generate_paper(cfg=cfg, **kwargs)


def paper_to_docx(paper: dict, figures: dict | None = None,
                  property_name: str = "dielectric") -> bytes:
    mod = _route(property_name)
    return mod.paper_to_docx(paper, figures=figures)


def generate_paper_figures(property_name: str = "dielectric", **kwargs) -> dict:
    mod = _route(property_name)
    return mod.generate_paper_figures(**kwargs)


def paper_to_latex(paper: dict, template: str = "generic",
                   author_info: dict | None = None,
                   property_name: str = "dielectric") -> bytes:
    """
    generate_paper() 결과 dict → LaTeX .tex bytes.

    template: "generic" | "elsevier" | "acs"
    현재는 dielectric 물성만 지원 (다른 물성 추가 시 각 모듈에 구현 필요).
    """
    mod = _route(property_name)
    if hasattr(mod, "paper_to_latex"):
        return mod.paper_to_latex(paper, template=template, author_info=author_info)
    # 폴백: 범용 모듈에 없으면 dielectric 구현을 직접 호출
    from . import _paper_dielectric as _dielectric_mod
    return _dielectric_mod.paper_to_latex(paper, template=template, author_info=author_info)
