# -*- coding: utf-8 -*-
"""
smoke_test.py — MIDAS_v02 컴포넌트 통합 검증 (CLI)

실행:  python smoke_test.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def banner(s: str):
    print("\n" + "=" * 70)
    print(f"  {s}")
    print("=" * 70)


def test_registry():
    banner("1. Property registry")
    from properties import list_properties, load_property
    props = list_properties()
    print(f"  Registered properties: {props}")
    assert "dielectric" in props, "dielectric must be registered"
    assert "solubility" in props, "solubility must be registered"

    for name in props:
        cfg = load_property(name)
        print(f"  ─ {name}: target={cfg.target_column}  "
              f"direction={cfg.direction}  log_tf={cfg.log_transform}  "
              f"threshold={cfg.screening_threshold}")
        assert cfg.dataset_path.exists(), f"{cfg.dataset_path} not found"


def test_dataset_load():
    banner("2. Dataset load + is_good mask")
    import pandas as pd
    from properties import load_property
    for name in ["dielectric", "solubility"]:
        cfg = load_property(name)
        df = pd.read_csv(cfg.dataset_path)
        assert cfg.target_column in df.columns, \
            f"{cfg.target_column} missing in {name} dataset"
        n_good = int(cfg.is_good(df[cfg.target_column]).sum())
        print(f"  {name}: {len(df)} rows, {n_good} pass threshold")


def test_feature_pipeline_dielectric():
    banner("3. Feature pipeline (dielectric, ~263 mols)")
    import pandas as pd
    from properties import load_property
    from engines.feature_engine import build_feature_matrix, select_features

    cfg = load_property("dielectric")
    df = pd.read_csv(cfg.dataset_path)
    X, y = build_feature_matrix(
        df, smiles_col=cfg.smiles_column,
        target_col=cfg.target_column, verbose=False,
    )
    print(f"  X shape: {X.shape}  y: min={y.min():.2f} max={y.max():.2f}")
    X_sel, names = select_features(X, y, n_features=20, verbose=False)
    print(f"  Top features: {names[:5]}")
    assert X_sel.shape[1] == 20


def test_feature_pipeline_solubility():
    banner("4. Feature pipeline (solubility, 50 mols)")
    import pandas as pd
    from properties import load_property
    from engines.feature_engine import build_feature_matrix, select_features

    cfg = load_property("solubility")
    df = pd.read_csv(cfg.dataset_path)
    X, y = build_feature_matrix(
        df, smiles_col=cfg.smiles_column,
        target_col=cfg.target_column, verbose=False,
    )
    print(f"  X shape: {X.shape}  y: min={y.min():.2f} max={y.max():.2f}")
    X_sel, names = select_features(X, y, n_features=15, verbose=False)
    print(f"  Top features: {names[:5]}")


def test_ml_pipeline_solubility():
    banner("5. ML pipeline (solubility, log_tf=False direction=higher_better)")
    import pandas as pd
    from properties import load_property
    from engines.feature_engine import build_feature_matrix, select_features
    from engines.ml_engine import run_pipeline_for_property

    cfg = load_property("solubility")
    df = pd.read_csv(cfg.dataset_path)
    X, y = build_feature_matrix(
        df, smiles_col=cfg.smiles_column,
        target_col=cfg.target_column, verbose=False,
    )
    X_sel, names = select_features(X, y, n_features=15, verbose=False)
    result = run_pipeline_for_property(X_sel, y, cfg, domain_mode=False, verbose=False)

    best = result["cv"]["best_model"]
    test_r2 = result["metrics"][best]["test"]["R2"]
    print(f"  best={best.upper()}  Test R²={test_r2:.3f}  "
          f"CV R²={result['cv']['r2_mean']:.3f}±{result['cv']['r2_std']:.3f}")


def test_screening_with_sort_direction():
    banner("6. Screening (solubility, higher_better → desc 정렬)")
    import pandas as pd
    from properties import load_property
    from engines.feature_engine import build_feature_matrix, select_features
    from engines.ml_engine import run_pipeline_for_property
    from engines.screening_engine import screen

    cfg = load_property("solubility")
    df = pd.read_csv(cfg.dataset_path)
    X, y = build_feature_matrix(
        df, smiles_col=cfg.smiles_column,
        target_col=cfg.target_column, verbose=False,
    )
    X_sel, names = select_features(X, y, n_features=15, verbose=False)
    result = run_pipeline_for_property(X_sel, y, cfg, verbose=False)
    X_tr = result["splits"]["X_tr"]

    smiles_text = (
        "Sugar,OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O\n"
        "Ethanol,CCO\n"
        "Octane,CCCCCCCC\n"
    )
    df_res = screen(smiles_text, names, result["models"], X_tr,
                    sort_ascending=cfg.sort_ascending())
    print(df_res[["rank", "name", "pred_mean", "ad_label"]].to_string(index=False))
    assert df_res.iloc[0]["pred_mean"] >= df_res.iloc[-1]["pred_mean"], \
        "higher_better 일 때 1위가 더 큰 pred_mean을 가져야 함"


def test_paper_router():
    banner("7. Paper router (generic template for solubility)")
    import pandas as pd
    from properties import load_property
    from engines.feature_engine import build_feature_matrix, select_features, get_feature_importances
    from engines.ml_engine import run_pipeline_for_property
    from engines.screening_engine import screen
    from engines.paper_engine import generate_paper

    cfg = load_property("solubility")
    df = pd.read_csv(cfg.dataset_path)
    X, y = build_feature_matrix(
        df, smiles_col=cfg.smiles_column,
        target_col=cfg.target_column, verbose=False,
    )
    X_sel, names = select_features(X, y, n_features=15, verbose=False)
    feat_imp = get_feature_importances(X_sel, y)
    result = run_pipeline_for_property(X_sel, y, cfg, verbose=False)
    X_tr = result["splits"]["X_tr"]
    df_scr = screen("Ethanol,CCO\nBenzene,c1ccccc1\n",
                    names, result["models"], X_tr,
                    sort_ascending=cfg.sort_ascending())

    paper = generate_paper(
        cfg=cfg,
        df_dataset=df, feat_names=names, X_all_shape=X.shape,
        cv_result=result["cv"], metrics=result["metrics"],
        df_screening=df_scr, feat_imp=feat_imp,
        log_tf=cfg.log_transform,
        best_model=result["cv"]["best_model"],
        model_mode=result.get("mode", "full"),
    )
    print(f"  Generated paper sections: {list(paper.keys())}")
    print(f"  Abstract preview: {paper['abstract'][:200]}...")
    assert "수용해도" in paper["abstract"]
    assert paper["full_md"].count("##") >= 5


def main():
    tests = [
        test_registry,
        test_dataset_load,
        test_feature_pipeline_dielectric,
        test_feature_pipeline_solubility,
        test_ml_pipeline_solubility,
        test_screening_with_sort_direction,
        test_paper_router,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS]")
        except Exception as e:
            failed += 1
            print(f"  [FAIL]: {e}")
            traceback.print_exc()
    print("\n" + "=" * 70)
    if failed == 0:
        print(f"  ALL {len(tests)} TESTS PASSED")
    else:
        print(f"  {failed} / {len(tests)} FAILED")
    print("=" * 70)
    return failed


if __name__ == "__main__":
    sys.exit(main())
