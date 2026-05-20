# -*- coding: utf-8 -*-
"""
properties/base.py
──────────────────
PropertyConfig — 모든 엔진이 의존하는 물성 메타데이터 단일 진실 공급원(SSOT).

v01의 하드코딩된 ``k_exp``, ``k < 2.4``, ``log(k)`` 같은 도메인 상수를
이 데이터클래스 하나로 일반화한다.  새 물성을 추가하려면
``properties/<name>/config.yaml`` 만 만들면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Direction = Literal["lower_better", "higher_better", "target_window"]
TaskType  = Literal["regression", "classification"]


@dataclass(frozen=True)
class PropertyConfig:
    # ── 식별 ──────────────────────────────────────────────────────────────
    name:           str           # 내부 식별자 (디렉토리명과 일치)
    display_name:   str           # UI 표시명
    domain:         str           # 응용 도메인 (논문 서론 컨텍스트)

    # ── 데이터셋 ──────────────────────────────────────────────────────────
    dataset_path:   Path          # CSV 절대 경로
    smiles_column:  str           # 분자식 컬럼명
    target_column:  str           # 예측 대상 컬럼명
    unit:           str           # 단위 (논문/UI 표시용)

    # ── 학습 설정 ─────────────────────────────────────────────────────────
    task_type:      TaskType      = "regression"
    log_transform:  bool          = True   # log(y) 변환 후 학습 여부
    stratify_bins:  tuple[float, ...] = (0.0,)  # split_data용 구간 경계

    # ── 스크리닝 기준 ─────────────────────────────────────────────────────
    direction:           Direction       = "lower_better"
    screening_threshold: float           = 0.0    # "이 값 너머가 우수"
    training_threshold:  float | None    = None   # 도메인 전용 모드 (None=미사용)

    # ── 도메인 힌트 (UI/논문용, 알고리즘에 영향 없음) ──────────────────────
    descriptor_emphasis: tuple[str, ...] = ()
    description_short:   str             = ""

    # ── 추가 메타 (논문 템플릿이 사용) ─────────────────────────────────────
    extra: dict = field(default_factory=dict)

    # ────────────────────────────────────────────────────────────────────
    def is_good(self, y) -> "np.ndarray":
        """direction과 threshold 기준으로 '우수' 마스크 반환."""
        import numpy as np
        y = np.asarray(y)
        if self.direction == "lower_better":
            return y < self.screening_threshold
        if self.direction == "higher_better":
            return y > self.screening_threshold
        # target_window: extra["window"] = (lo, hi)
        lo, hi = self.extra.get("window", (-np.inf, np.inf))
        return (y >= lo) & (y <= hi)

    def sort_ascending(self) -> bool:
        """스크리닝 결과 정렬 방향 — '좋은 것'이 위로."""
        return self.direction == "lower_better"
