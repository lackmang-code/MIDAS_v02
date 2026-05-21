# -*- coding: utf-8 -*-
"""
MIDAS_v02  —  Multi-Property QSPR Platform
==========================================
사이드바: 타겟 물성 선택  →  모든 탭(S1~S5)이 그 물성을 기준으로 동작.

탭 구성
  S1. Dataset       — EDA, 분포, 우수후보 테이블
  S2. Features      — RDKit 233개 → Top-N 선택
  S3. ML Model      — GBR/RF/Ridge/GPR 학습 + CV + AD
  S4. Screening     — 후보 분자 예측 + AD 판정
  S5. Paper         — 도메인-라우팅 논문 초안 자동 생성

v01 대비 변경점
  - 데이터셋·타겟·로그변환·정렬방향이 PropertyConfig 로 일반화
  - 사이드바에서 "유전율 / 수용해도 / ..." 선택 가능
  - 새 물성 추가 = properties/<name>/config.yaml + dataset.csv 만들면 끝
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from rdkit import Chem

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from properties import list_properties, load_property, PropertyConfig
from engines.feature_engine import (
    build_feature_matrix, select_features, get_feature_importances
)
from engines.ml_engine import (
    split_data, train_all_models, cross_validate, auto_select_best,
    evaluate, predict_ensemble, calc_leverage, run_pipeline_for_property
)
from engines.screening_engine import (
    screen, load_example_candidates, draw_grid, mol_to_png_bytes
)
from engines.paper_engine import (
    generate_paper, paper_to_docx, generate_paper_figures, paper_to_latex
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
N_FEATURES = 50


# ═════════════════════════════════════════════════════════════════════════════
# 상태 관리 — 물성을 바꾸면 캐시 초기화
# ═════════════════════════════════════════════════════════════════════════════
def _reset_caches():
    for k in ["df", "X_all", "X_sel", "y", "feat_names", "feat_imp",
              "ml_result", "s4_result", "paper", "paper_figures",
              "paper_saved_paths",
              "s1_done", "s2_done", "s3_done", "s4_done", "s5_done"]:
        st.session_state.pop(k, None)


def _current_property() -> PropertyConfig:
    return st.session_state["property_cfg"]


def load_data() -> pd.DataFrame:
    if "df" not in st.session_state:
        cfg = _current_property()
        df = pd.read_csv(cfg.dataset_path)
        st.session_state["df"] = df
    return st.session_state["df"]


def _fig_to_st(fig, caption: str = ""):
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# S1 — Dataset EDA  (물성-무관)
# ═════════════════════════════════════════════════════════════════════════════
def render_s1():
    cfg = _current_property()
    target = cfg.target_column

    st.subheader(f"📊 Dataset EDA — {cfg.display_name}")
    df = load_data()

    if target not in df.columns:
        st.error(f"❌ 데이터셋에 타겟 컬럼 '{target}' 이 없습니다. "
                 f"config.yaml 의 target_column 또는 dataset.csv 를 확인하세요.")
        st.dataframe(df.head())
        return

    # ── 통계 카드 ─────────────────────────────────────────────────────────────
    y = df[target]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total molecules", f"{len(df)}")
    c2.metric(f"{target} range", f"{y.min():.2f} ~ {y.max():.2f}")
    c3.metric("Mean / Median", f"{y.mean():.2f} / {y.median():.2f}")
    c4.metric(f"기준 만족 (good)", f"{int(cfg.is_good(y).sum())}개")

    st.caption(f"기준: {cfg.target_column} "
               f"{'<' if cfg.direction=='lower_better' else '>'} "
               f"{cfg.screening_threshold:g}  ({cfg.unit})")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(f"**{target} 분포 (전체)**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(y, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(cfg.screening_threshold, color="red", ls="--", lw=1.5,
                   label=f"threshold = {cfg.screening_threshold:g}")
        ax.set_xlabel(f"{target} ({cfg.unit})")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {target}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        _fig_to_st(fig)

    with col_r:
        if "category" in df.columns:
            st.markdown(f"**{target} by Category**")
            cats = df.groupby("category")[target].median().sort_values()
            ordered = cats.index.tolist()
            data_by = [df[df["category"] == c][target].values for c in ordered]

            fig, ax = plt.subplots(figsize=(5, 3.5))
            bp = ax.boxplot(data_by, patch_artist=True,
                            medianprops=dict(color="black", lw=2))
            colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(ordered)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(ordered) + 1))
            ax.set_xticklabels(ordered, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel(target)
            ax.axhline(cfg.screening_threshold, color="red", ls="--", lw=1)
            plt.tight_layout()
            _fig_to_st(fig)
        else:
            st.info("`category` 컬럼이 없어 박스플롯 생략")

    # ── 우수 후보 테이블 ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"**우수 후보  (기준 만족)**")
    good_mask = cfg.is_good(y)
    sort_asc = cfg.sort_ascending()
    show_cols = [c for c in ["name", target, "category", "source"]
                 if c in df.columns]
    good = (df[good_mask].sort_values(target, ascending=sort_asc)[show_cols]
            .reset_index(drop=True))
    good.index += 1
    st.dataframe(good, use_container_width=True, height=350)

    if "source" in df.columns:
        st.markdown("---")
        st.markdown("**Source breakdown**")
        st.dataframe(df["source"].value_counts().rename("count"),
                     use_container_width=True)

    st.session_state["s1_done"] = True


# ═════════════════════════════════════════════════════════════════════════════
# S2 — Feature Engineering  (물성-무관)
# ═════════════════════════════════════════════════════════════════════════════
def render_s2():
    cfg = _current_property()
    st.subheader(f"🔬 Feature Engineering — {cfg.display_name}")

    df = load_data()

    if "X_sel" in st.session_state:
        st.success(f"✅ 피처 캐시됨  ({st.session_state['X_sel'].shape[1]}개 선택 / "
                   f"{st.session_state['X_all'].shape[0]}개 분자)")
        if not st.button("🔄 피처 재계산"):
            _render_s2_plots()
            return
        for k in ["X_all", "y", "X_sel", "feat_names", "feat_imp"]:
            st.session_state.pop(k, None)

    st.info(f"RDKit 기술자 계산 중... ({len(df)}개 분자)")
    prog = st.progress(0, text="계산 중...")

    with st.spinner("분자 기술자 계산 중..."):
        X_all, y = build_feature_matrix(
            df,
            smiles_col=cfg.smiles_column,
            target_col=cfg.target_column,
            verbose=False,
        )
        prog.progress(40)

    with st.spinner("피처 선택 중..."):
        X_sel, feat_names = select_features(X_all, y, n_features=N_FEATURES, verbose=False)
        prog.progress(80)

    with st.spinner("중요도 계산 중..."):
        feat_imp = get_feature_importances(X_sel, y)
        prog.progress(100)

    st.session_state.update({
        "X_all": X_all, "y": y, "X_sel": X_sel,
        "feat_names": feat_names, "feat_imp": feat_imp,
    })
    prog.empty()
    st.success(f"✅ 피처 계산 완료  ({X_all.shape[1]}개 원시 → {X_sel.shape[1]}개 선택)")
    _render_s2_plots()


def _render_s2_plots():
    cfg = _current_property()
    X_all    = st.session_state["X_all"]
    y        = st.session_state["y"]
    X_sel    = st.session_state["X_sel"]
    feat_imp = st.session_state["feat_imp"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Raw features",       f"{X_all.shape[1]}")
    c2.metric("Pre-RF filter",      f"~{X_sel.shape[1]+20}")
    c3.metric("Final selected",     f"{X_sel.shape[1]}")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("**Top-20 Feature Importances (RF-based)**")
        top20 = feat_imp.head(20)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top20)))[::-1]
        ax.barh(range(len(top20)), top20.values[::-1],
                color=colors, edgecolor="white")
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20.index[::-1], fontsize=8)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        _fig_to_st(fig)

    with col_r:
        st.markdown("**Top-10 Features**")
        top10_df = pd.DataFrame({
            "Feature":    feat_imp.head(10).index,
            "Importance": feat_imp.head(10).values.round(4),
        })
        st.dataframe(top10_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(f"**물성별 강조 기술자 (config.yaml)**")
        if cfg.descriptor_emphasis:
            st.write(", ".join(cfg.descriptor_emphasis))
        else:
            st.caption("— 별도 강조 기술자 없음 —")

    # ── 핵심 피처 vs target 산점도 ────────────────────────────────────────────
    st.markdown("---")
    emphasis = [f for f in cfg.descriptor_emphasis if f in X_all.columns][:3]
    if not emphasis:
        emphasis = [f for f in feat_imp.head(3).index if f in X_all.columns]
    if emphasis:
        st.markdown(f"**Key features vs {cfg.target_column}**")
        fig, axes = plt.subplots(1, len(emphasis), figsize=(5 * len(emphasis), 3.8))
        if len(emphasis) == 1:
            axes = [axes]
        good_mask = cfg.is_good(y)
        for ax, feat in zip(axes, emphasis):
            colors_sc = np.where(good_mask, "#27ae60", "#e74c3c")
            ax.scatter(X_all[feat], y, c=colors_sc, s=20, alpha=0.7, edgecolors="none")
            ax.set_xlabel(feat)
            ax.set_ylabel(cfg.target_column)
            ax.set_title(f"{feat} vs {cfg.target_column}")
            ax.axhline(cfg.screening_threshold, color="red", ls="--", lw=1)
        plt.tight_layout()
        _fig_to_st(fig, "초록 = 기준 만족 분자")

    st.session_state["s2_done"] = True


# ═════════════════════════════════════════════════════════════════════════════
# S3 — ML Model Training
# ═════════════════════════════════════════════════════════════════════════════
def render_s3():
    cfg = _current_property()
    st.subheader(f"🤖 ML Model Training — {cfg.display_name}")

    if "X_sel" not in st.session_state:
        st.warning("⚠️ S2 탭에서 피처를 먼저 계산하세요.")
        return

    X_sel = st.session_state["X_sel"]
    y     = st.session_state["y"]

    if "ml_result" in st.session_state:
        st.success("✅ 모델 학습 결과 캐시됨")
        if not st.button("🔄 재학습"):
            _render_s3_results()
            return
        st.session_state.pop("ml_result", None)

    st.markdown("**학습 설정**")
    c1, c2 = st.columns(2)
    log_default = bool(cfg.log_transform)
    log_tf = c1.checkbox(f"log({cfg.target_column}) 변환 학습",
                         value=log_default,
                         help="PropertyConfig 기본값을 따른다")
    domain_only = False
    if cfg.training_threshold is not None:
        op = "<" if cfg.direction == "lower_better" else ">"
        domain_only = c2.checkbox(
            f"도메인 전용 모드 ({cfg.target_column} {op} {cfg.training_threshold:g})",
            value=False,
            help="우수 영역만 학습 (예: low-k 전용)"
        )

    if st.button("▶ 모델 학습 시작", type="primary"):
        prog = st.progress(0, text="파이프라인 실행 중...")
        with st.spinner("학습 중 (20-40초)..."):
            # PropertyConfig 기반 파이프라인 사용
            # log_tf 는 사용자가 UI에서 바꿀 수 있으므로 cfg 복사본을 사용
            from dataclasses import replace
            cfg_run = replace(cfg, log_transform=log_tf)
            prog.progress(20, text="분할 + 학습 + CV...")
            result = run_pipeline_for_property(
                X_sel, y, cfg_run,
                domain_mode=domain_only,
                verbose=False,
            )
            prog.progress(80, text="AD 계산 중...")
            X_tr = result["splits"]["X_tr"]
            X_te = result["splits"]["X_te"]
            h_te, h_star = calc_leverage(X_tr, X_te)
            result["leverage"] = {"h_te": h_te, "h_star": h_star}
            result["log_tf"]   = log_tf
            result["n_train"]  = len(X_tr)
            prog.progress(100, text="완료!")

        st.session_state["ml_result"] = result
        prog.empty()
        st.success("✅ 학습 완료!")
        _render_s3_results()


def _render_s3_results():
    cfg  = _current_property()
    res  = st.session_state["ml_result"]
    mets = res["metrics"]
    cv   = res["cv"]

    st.markdown("---")
    st.markdown("### 모델 성능 비교")
    rows = []
    for mname in [m for m in ["gbr", "rf", "ridge", "gpr"] if m in mets]:
        for split in ["train", "val", "test"]:
            m = mets[mname][split]
            rows.append({
                "Model":  mname.upper(),
                "Split":  split,
                "RMSE":   m["RMSE"],
                "MAE":    m["MAE"],
                "R²":     m["R2"],
                "logRMSE": m.get("log_RMSE", "-"),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 5-Fold CV (Best model)")
    c1, c2, c3 = st.columns(3)
    c1.metric("CV RMSE", f"{cv['rmse_mean']:.3f} ± {cv['rmse_std']:.3f}")
    c2.metric("CV MAE",  f"{cv['mae_mean']:.3f}")
    c3.metric("CV R²",   f"{cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}")

    # Actual vs Predicted
    st.markdown("---")
    best = cv.get("best_model", "gbr")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(f"**Actual vs Predicted — {best.upper()} (Test)**")
        y_true = mets[best]["test"]["y_true"]
        y_pred = mets[best]["test"]["y_pred"]
        good_mask = cfg.is_good(y_true)
        fig, ax = plt.subplots(figsize=(5, 4))
        colors_sc = np.where(good_mask, "#27ae60", "#e74c3c")
        ax.scatter(y_true, y_pred, c=colors_sc, s=40, alpha=0.85, edgecolors="white", lw=0.5)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
        ax.set_xlabel(f"Actual {cfg.target_column}")
        ax.set_ylabel(f"Predicted {cfg.target_column}")
        ax.set_title(f"{best.upper()} Test R²={mets[best]['test']['R2']:.3f}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        _fig_to_st(fig)

    with col_r:
        st.markdown("**5-Fold CV R² per fold**")
        folds = range(1, 6)
        r2s = cv["r2_folds"]
        colors_cv = ["#27ae60" if v >= 0.55 else "#f39c12" if v >= 0.40 else "#e74c3c"
                     for v in r2s]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(folds, r2s, color=colors_cv, edgecolor="white", alpha=0.9)
        ax.axhline(cv["r2_mean"], color="navy", ls="--", lw=1.5,
                   label=f"Mean = {cv['r2_mean']:.3f}")
        ax.set_xticks(list(folds))
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        plt.tight_layout()
        _fig_to_st(fig)

    # Williams plot
    st.markdown("---")
    h_te   = res["leverage"]["h_te"]
    h_star = res["leverage"]["h_star"]
    y_true = mets[best]["test"]["y_true"]
    y_pred = mets[best]["test"]["y_pred"]
    rmse_te = mets[best]["test"]["RMSE"]
    std_res = (y_pred - y_true) / rmse_te if rmse_te > 0 else np.zeros_like(y_pred)
    colors_ad = np.where(h_te <= h_star, "#3498db", "#e74c3c")

    st.markdown("**Williams Plot (Applicability Domain)**")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.scatter(h_te, std_res, c=colors_ad, s=35, alpha=0.85, edgecolors="white", lw=0.4)
    ax.axvline(h_star, color="red",    ls="--", lw=1.5, label=f"h* = {h_star:.3f}")
    ax.axhline(3.0,    color="orange", ls="--", lw=1, label="±3σ")
    ax.axhline(-3.0,   color="orange", ls="--", lw=1)
    ax.set_xlabel("Leverage h")
    ax.set_ylabel("Standardized Residual")
    n_in = int((h_te <= h_star).sum()); n_out = int((h_te > h_star).sum())
    ax.legend(fontsize=8, title=f"AD inside:{n_in}  outside:{n_out}")
    plt.tight_layout()
    _fig_to_st(fig)

    st.success(f"**최적 모델: {best.upper()}**  |  "
               f"Test R²={mets[best]['test']['R2']:.3f}  "
               f"CV R²={cv['r2_mean']:.3f}±{cv['r2_std']:.3f}")
    st.session_state["s3_done"] = True


# ═════════════════════════════════════════════════════════════════════════════
# S4 — Screening
# ═════════════════════════════════════════════════════════════════════════════
def render_s4():
    cfg = _current_property()
    st.subheader(f"🏆 Candidate Screening — {cfg.display_name}")

    if "ml_result" not in st.session_state:
        st.warning("⚠️ S3 탭에서 모델을 먼저 학습하세요.")
        return
    if "feat_names" not in st.session_state:
        st.warning("⚠️ S2 탭에서 피처를 먼저 계산하세요.")
        return

    models     = st.session_state["ml_result"]["models"]
    X_tr       = st.session_state["ml_result"]["splits"]["X_tr"]
    feat_names = st.session_state["feat_names"]

    st.markdown("**입력 방식 선택**")
    mode = st.radio("", ["📋 내장 후보 목록 사용", "✏️ 직접 SMILES 입력"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "📋 내장 후보 목록 사용":
        candidates = load_example_candidates(cfg.name)
        st.info(f"내장 후보 {len(candidates)}개 로드됨 (물성: {cfg.name})")
        default_text = "\n".join(f"{n},{s}" for n, s in candidates)
    else:
        default_text = (
            "# Name,SMILES 형식으로 입력 (한 줄씩)\n"
            "Ethanol,CCO\n"
            "Hexafluorobenzene,Fc1c(F)c(F)c(F)c(F)c1F\n"
        )

    smiles_text = st.text_area("SMILES 입력", value=default_text, height=200)

    if st.button("🔍 스크리닝 실행", type="primary"):
        with st.spinner("기술자 계산 및 예측 중..."):
            df_result = screen(
                smiles_text, feat_names, models, X_tr,
                sort_ascending=cfg.sort_ascending(),
            )
        st.session_state["s4_result"] = df_result
        st.success(f"✅ 완료! 유효 분자: {int(df_result['valid'].sum())} / {len(df_result)}")

    if "s4_result" not in st.session_state:
        return

    df_result = st.session_state["s4_result"]
    df_valid  = df_result[df_result["valid"] == True].copy()
    if df_valid.empty:
        st.error("유효한 SMILES가 없습니다.")
        return

    st.markdown("---")
    good_pred = cfg.is_good(df_valid["pred_mean"])
    c1, c2, c3 = st.columns(3)
    c1.metric("스크리닝 후보", f"{len(df_valid)}개")
    c2.metric(f"기준 만족 예측", f"{int(good_pred.sum())}개")
    c3.metric("AD 내부", f"{int(df_valid['ad_ok'].sum())}개")

    st.markdown("---")
    t1, t2, t3 = st.tabs(["📋 결과", "🧪 구조", "📊 차트"])

    with t1:
        st.markdown(f"**예측 결과  (정렬: pred_mean "
                    f"{'asc' if cfg.sort_ascending() else 'desc'})**")
        cols = ["rank", "name", "pred_mean", "pred_std",
                "pred_gbr", "pred_rf", "pred_ridge", "h", "ad_label"]
        cols = [c for c in cols if c in df_valid.columns]
        st.dataframe(df_valid[cols], use_container_width=True, hide_index=True)
        csv = df_valid.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇ CSV 다운로드", csv,
                           f"screening_{cfg.name}.csv", "text/csv")

    with t2:
        top20 = df_valid.head(20)
        grid_bytes = draw_grid(
            top20["smiles"].tolist(), top20["name"].tolist(),
            top20["pred_mean"].tolist(), n_cols=4,
        )
        if grid_bytes:
            st.image(grid_bytes, use_container_width=True)

    with t3:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        ax.hist(df_valid["pred_mean"], bins=20,
                color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(cfg.screening_threshold, color="red", ls="--", lw=1.5,
                   label=f"thresh = {cfg.screening_threshold:g}")
        ax.set_xlabel(f"Predicted {cfg.target_column}")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        colors_ad = np.where(df_valid["ad_ok"], "#27ae60", "#e74c3c")
        ax2.scatter(df_valid["pred_mean"], df_valid["pred_std"],
                    c=colors_ad, s=60, alpha=0.85, edgecolors="white", lw=0.5)
        ax2.set_xlabel(f"Predicted {cfg.target_column}")
        ax2.set_ylabel("Uncertainty (std)")
        plt.tight_layout()
        _fig_to_st(fig)

    st.session_state["s4_done"] = True


# ═════════════════════════════════════════════════════════════════════════════
# S5 — Paper Draft  (라우터 기반)
# ═════════════════════════════════════════════════════════════════════════════
def render_s5():
    from datetime import datetime as _dt
    cfg = _current_property()
    st.subheader(f"📄 Paper Draft — {cfg.display_name}")

    missing = []
    if "ml_result"  not in st.session_state: missing.append("S3 ML 학습")
    if "feat_names" not in st.session_state: missing.append("S2 피처 계산")
    if "s4_result"  not in st.session_state: missing.append("S4 스크리닝")
    if missing:
        st.warning(f"⚠️ 먼저 완료하세요: {', '.join(missing)}")
        return

    df = load_data()
    res = st.session_state["ml_result"]
    feat_imp = st.session_state["feat_imp"]

    if "paper" not in st.session_state:
        # ── 저자 정보 입력 UI ─────────────────────────────────────────────────
        with st.expander("👤 저자 정보 (논문에 자동 삽입)", expanded=True):
            col1, col2 = st.columns(2)
            col1.text_input("저자명", key="s5_author_name",
                            placeholder="홍길동, 김철수")
            col1.text_input("소속기관", key="s5_author_affil",
                            placeholder="성균관대학교 첨단디스플레이공학과")
            col2.text_input("교신저자 이메일", key="s5_author_email",
                            placeholder="author@skku.edu")
            col2.text_input("ORCID", key="s5_author_orcid",
                            placeholder="0000-0000-0000-0000")
            st.text_area(
                "감사의 말씀 (연구비 지원 기관, 과제번호 포함)",
                key="s5_ack", height=60,
                placeholder="본 연구는 [기관명] [과제번호]의 지원을 받아 수행되었습니다.",
            )

        if st.button("📝 논문 초안 생성", type="primary"):
            author_info = {
                "name":        st.session_state.get("s5_author_name", ""),
                "affiliation": st.session_state.get("s5_author_affil", ""),
                "email":       st.session_state.get("s5_author_email", ""),
                "orcid":       st.session_state.get("s5_author_orcid", ""),
            }
            with st.spinner("논문 초안 생성 중..."):
                paper = generate_paper(
                    cfg=cfg,
                    df_dataset    = df,
                    feat_names    = st.session_state["feat_names"],
                    X_all_shape   = st.session_state["X_all"].shape,
                    cv_result     = res["cv"],
                    metrics       = res["metrics"],
                    df_screening  = st.session_state["s4_result"],
                    feat_imp      = feat_imp,
                    log_tf        = res.get("log_tf", cfg.log_transform),
                    best_model    = res.get("cv", {}).get("best_model", "gbr"),
                    model_mode    = res.get("mode", "full"),
                    author_info   = author_info,
                    acknowledgments = st.session_state.get("s5_ack", ""),
                )
                st.session_state["paper"] = paper

                # 논문 삽입용 그림 생성 (Fig1~6) — 실패해도 논문은 진행
                try:
                    figures = generate_paper_figures(
                        property_name = cfg.name,
                        feat_imp      = feat_imp,
                        metrics       = res["metrics"],
                        df_screening  = st.session_state["s4_result"],
                        k_threshold   = cfg.screening_threshold,
                        best_model    = res.get("cv", {}).get("best_model", "gbr"),
                        splits        = res.get("splits"),
                    )
                except Exception as e:
                    figures = {}
                    st.warning(f"⚠️ 그림 생성 실패 (논문은 그림 없이 진행): {e}")
                st.session_state["paper_figures"] = figures

                # 자동 저장 — MD + DOCX + TEX 세 형식 모두
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                now_str   = _dt.now().strftime("%Y%m%d_%H%M%S")
                base_path = os.path.join(OUTPUT_DIR, f"{cfg.name}_paper_{now_str}")
                saved_paths = {}

                # Markdown
                md_path = base_path + ".md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(paper["full_md"])
                saved_paths["Markdown"] = md_path

                # Word (.docx)
                try:
                    docx_data = paper_to_docx(paper, figures=figures,
                                              property_name=cfg.name)
                    docx_path = base_path + ".docx"
                    with open(docx_path, "wb") as f:
                        f.write(docx_data)
                    saved_paths["Word"] = docx_path
                except Exception as e:
                    saved_paths["Word"] = f"저장 실패: {e}"

                # LaTeX (.tex)
                try:
                    latex_tmpl = st.session_state.get("s5_latex_template", "generic")
                    tex_data = paper_to_latex(
                        paper, template=latex_tmpl, author_info=author_info)
                    tex_path = base_path + ".tex"
                    with open(tex_path, "wb") as f:
                        f.write(tex_data)
                    saved_paths["LaTeX"] = tex_path
                except Exception as e:
                    saved_paths["LaTeX"] = f"저장 실패: {e}"

                st.session_state["paper_saved_paths"] = saved_paths

            # 메타 캡션 (논문 본문 밖 표시)
            res_cv   = res.get("cv", {})
            bm_label = res_cv.get("best_model", "gbr").upper()
            cv_r2    = res_cv.get("r2_mean", 0.0)
            n_total  = len(df)
            mode_lbl = res.get("mode", "full")
            st.caption(
                f"생성일: {date.today()}  |  데이터: {n_total}종  |  "
                f"최적 모델: {bm_label}  |  모드: {mode_lbl}  |  CV R²: {cv_r2:.3f}"
            )
            st.success("✅ 논문 초안 생성 완료!")
        return

    paper = st.session_state["paper"]
    saved = st.session_state.get("paper_saved_paths", {})
    if saved:
        with st.expander("💾 자동 저장 위치", expanded=False):
            for label, path in saved.items():
                st.markdown(f"- **{label}**: `{path}`")

    if st.button("🔄 재생성"):
        st.session_state.pop("paper", None)
        st.rerun()

    tabs = st.tabs(["📋 Abstract", "📖 Intro", "🔧 Methods",
                    "📊 Results", "🏁 Conclusion", "⚠️ Limitations",
                    "📄 전체 MD"])
    sections = ["abstract", "intro", "methods", "results",
                "conclusion", "limitations", "full_md"]
    for tab, sec in zip(tabs, sections):
        with tab:
            if sec == "full_md":
                edited = st.text_area(
                    "편집 (Markdown)",
                    value=st.session_state["paper"].get(sec, ""),
                    height=600,
                    key=f"s5_edit_{sec}",
                )
                if edited != st.session_state["paper"].get(sec, ""):
                    st.session_state["paper"][sec] = edited
            else:
                edited = st.text_area(
                    "편집 (Markdown)",
                    value=st.session_state["paper"].get(sec, ""),
                    height=500,
                    key=f"s5_edit_{sec}",
                )
                if edited != st.session_state["paper"].get(sec, ""):
                    st.session_state["paper"][sec] = edited
                with st.expander("📖 렌더링 미리보기", expanded=False):
                    st.markdown(edited)

    # 논문 통계
    word_count = len(paper.get("full_md", "").split())
    char_count = len(paper.get("full_md", ""))
    st.caption(f"📊 논문 통계: 단어 {word_count:,}개 · 문자 {char_count:,}자")

    st.markdown("---")

    # 저자 정보 수집
    ai = {
        "name":        st.session_state.get("s5_author_name", ""),
        "affiliation": st.session_state.get("s5_author_affil", ""),
        "email":       st.session_state.get("s5_author_email", ""),
        "orcid":       st.session_state.get("s5_author_orcid", ""),
    }

    st.markdown("#### 다운로드")
    col_d1, col_d2, col_d3 = st.columns(3)

    docx_bytes = paper_to_docx(paper,
                               figures=st.session_state.get("paper_figures", {}),
                               property_name=cfg.name)
    col_d1.download_button(
        "⬇ Word (.docx)", data=docx_bytes,
        file_name=f"{cfg.name}_paper.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    col_d2.download_button(
        "⬇ Markdown (.md)",
        data=paper["full_md"].encode("utf-8"),
        file_name=f"{cfg.name}_paper.md", mime="text/markdown",
    )

    # LaTeX 다운로드
    latex_template = st.session_state.get("s5_latex_template", "generic")
    latex_bytes = paper_to_latex(
        paper, template=latex_template,
        author_info=ai, property_name=cfg.name,
    )
    col_d3.download_button(
        "⬇ LaTeX (.tex)", data=latex_bytes,
        file_name=f"{cfg.name}_paper.tex",
        mime="text/plain",
    )

    # LaTeX 템플릿 선택 (다운로드 버튼 아래)
    st.radio(
        "LaTeX 저널 템플릿",
        options=["generic", "elsevier", "acs"],
        format_func=lambda x: {
            "generic":  "Generic (Article)",
            "elsevier": "Elsevier (elsarticle)",
            "acs":      "ACS (achemso)",
        }[x],
        key="s5_latex_template",
        horizontal=True,
    )

    st.session_state["s5_done"] = True


# ═════════════════════════════════════════════════════════════════════════════
# 메인
# ═════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="MIDAS v02 — Multi-Property QSPR",
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── 사이드바 — 물성 선택 ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚗️ MIDAS v02")
        st.caption("Multi-Property QSPR Platform")
        st.markdown("---")

        available = list_properties()
        if not available:
            st.error("등록된 물성이 없습니다. `properties/<name>/config.yaml` 추가 필요.")
            return

        # 현재 선택 유지
        cur_name = st.session_state.get("property_name", available[0])
        if cur_name not in available:
            cur_name = available[0]

        chosen = st.selectbox(
            "🎯 타겟 물성",
            options=available,
            index=available.index(cur_name),
            format_func=lambda n: load_property(n).display_name,
        )

        # 변경 감지 → 캐시 초기화
        if chosen != cur_name:
            _reset_caches()
            st.session_state["property_name"] = chosen

        st.session_state["property_name"] = chosen
        cfg = load_property(chosen)
        st.session_state["property_cfg"] = cfg

        st.markdown(f"**Domain:** {cfg.domain}")
        st.caption(cfg.description_short)
        st.markdown("---")
        st.markdown("**설정 요약**")
        st.markdown(
            f"- target: `{cfg.target_column}` ({cfg.unit})\n"
            f"- direction: `{cfg.direction}`\n"
            f"- threshold: `{cfg.screening_threshold:g}`\n"
            f"- log-transform: `{cfg.log_transform}`\n"
            f"- task: `{cfg.task_type}`"
        )

    # ── 메인 화면 ────────────────────────────────────────────────────────────
    st.markdown(
        f"<h2 style='margin-bottom:0'>⚗️ MIDAS v02 — {cfg.display_name}</h2>"
        f"<p style='color:gray;margin-top:4px'>{cfg.domain}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    s_ok = lambda k: "✅" if st.session_state.get(k) else "○"
    st.caption(
        f"{s_ok('s1_done')} S1 Dataset  →  "
        f"{s_ok('s2_done')} S2 Features  →  "
        f"{s_ok('s3_done')} S3 ML Model  →  "
        f"{s_ok('s4_done')} S4 Screening  →  "
        f"{s_ok('s5_done')} S5 Paper"
    )

    tabs = st.tabs([
        "📊 S1. Dataset",
        "🔬 S2. Features",
        "🤖 S3. ML Model",
        "🏆 S4. Screening",
        "📄 S5. Paper",
    ])
    with tabs[0]: render_s1()
    with tabs[1]: render_s2()
    with tabs[2]: render_s3()
    with tabs[3]: render_s4()
    with tabs[4]: render_s5()


if __name__ == "__main__":
    main()
