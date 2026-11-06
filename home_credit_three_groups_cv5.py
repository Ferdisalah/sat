#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Home Credit: CV5 validation of the selected combinations from groups 1, 2, 3.

Design
------
* 32 original raw predictors are retained unchanged in every fit.
* CREDIT_TERM is fixed in every fit.
* The 15 combinations from Group 1, 10 from Group 2, and 15 from Group 3
  are consolidated by an order-independent feature key.
* Five combinations shared by Groups 2 and 3 are evaluated only once.
* A raw+ CREDIT_TERM baseline is added as a reference.
* Result: 35 unique candidates + 1 baseline = 36 combinations; 180 CV fits.
* Stratified 5-fold splits are identical for every candidate.
* Target-informed OCCUPATION_TOP3_GROUP is selected using each fold's
  training labels only. Region income means are also fit on each fold's
  training rows only.
* Fold predictions and metrics are checkpointed after every fit so the run
  can resume after interruption.
* After CV, the top-ranked combination is refit on all labeled rows and a
  test submission is generated, following the previous script's output style.

Run from Windows:
    %runfile C:/Users/UNITECH/Downloads/home_credit_three_groups_cv5.py --wdir

Required input files are the same Windows paths used in the supplied script.
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations  # retained for clarity if candidate plans are extended
from time import perf_counter
import gc
import hashlib
import json
import os
import zlib

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


# ============================================================================
# 1. CONFIGURATION
# ============================================================================
TRAIN_FILE = Path(r"C:\Users\UNITECH\Downloads\train.csv")
TEST_FILE = Path(r"C:\Users\UNITECH\Downloads\test (1).csv")

OUTPUT_DIR = Path(r"C:\Users\UNITECH\Desktop\GCI 2026\competition\output")

TASK_TYPE = "GPU"  # Keep GPU for comparability with the preceding experiments.
RANDOM_SEED = 42
N_SPLITS = 5

MAX_ITERATIONS = 3000
LEARNING_RATE = 0.04
DEPTH = 6
EARLY_STOPPING_ROUNDS = 200
LOG_PERIOD = 200
L2_LEAF_REG = 15

SUBGROUP_GOODS_PRICE = 450000
MIN_CATEGORY_COUNT = 25
TOP_INCOME_TYPES = [
    "Working",
    "Commercial associate",
    "State servant",
    "Pensioner",
]

SCRIPT_VERSION = "three_groups_cv5_v1.0"
MISSING_TOKEN = "__MISSING__"

# Resume checkpoints. These are intermediate files, not final model artifacts.
CHECKPOINT_DIR = OUTPUT_DIR / "cv5_three_groups_checkpoints"

# Distinct output names: previous reports and models are not overwritten.
FOLD_REPORT_FILE = OUTPUT_DIR / f"cv5_three_groups_fold_metrics_{TASK_TYPE.lower()}.csv"
SUMMARY_REPORT_FILE = OUTPUT_DIR / f"cv5_three_groups_summary_{TASK_TYPE.lower()}.csv"
OOF_PREDICTIONS_FILE = OUTPUT_DIR / f"cv5_three_groups_oof_predictions_{TASK_TYPE.lower()}.csv.gz"
BEST_MODEL_FILE = OUTPUT_DIR / f"best_catboost_three_groups_cv5_{TASK_TYPE.lower()}.cbm"
BEST_FEATURES_FILE = OUTPUT_DIR / f"best_feature_subset_three_groups_cv5_{TASK_TYPE.lower()}.json"
BEST_SUBMISSION_FILE = OUTPUT_DIR / f"submission_best_three_groups_cv5_{TASK_TYPE.lower()}.csv"
BEST_IMPORTANCE_FILE = OUTPUT_DIR / f"best_model_feature_importance_three_groups_cv5_{TASK_TYPE.lower()}.csv"


# ============================================================================
# 2. ENGINEERED FEATURE DEFINITIONS
# ============================================================================
FIXED_ENGINEERED_FEATURES = ["CREDIT_TERM"]

OLD_OPTIONAL_FEATURES = [
    "extr2",
    "ID_PUBLISH_AGE",
    "REGISTRATION_AGE",
    "PHONE_CHANGE_AGE",
    "PHONE_CHANGE_ZERO",
    "AGE_YEARS",
    "OCCUPATION_TOP3_GROUP",
    "INCOME_TYPE_TOP4_GROUP",
]

NEW_OPTIONAL_FEATURES = [
    "INCOME_TO_REGION_MEAN",
    "INCOME_GAP_FROM_REGION_MEAN",
    "WORKING_CASH_INDICATOR",
    "WORKING_CASH_EMPLOYMENT_YEARS",
]

ALL_OPTIONAL_FEATURES = OLD_OPTIONAL_FEATURES + NEW_OPTIONAL_FEATURES
FEATURE_ORDER = {name: i for i, name in enumerate(ALL_OPTIONAL_FEATURES)}

CATEGORICAL_ENGINEERED_FEATURES = {
    "OCCUPATION_TOP3_GROUP",
    "INCOME_TYPE_TOP4_GROUP",
}


# ============================================================================
# 3. THE THREE CANDIDATE GROUPS
#    CREDIT_TERM is deliberately omitted here because it is fixed in all fits.
# ============================================================================
GROUP_1 = [
    "extr2; PHONE_CHANGE_AGE; PHONE_CHANGE_ZERO; WORKING_CASH_INDICATOR; WORKING_CASH_EMPLOYMENT_YEARS",
    "AGE_YEARS; WORKING_CASH_EMPLOYMENT_YEARS",
    "OCCUPATION_TOP3_GROUP; WORKING_CASH_EMPLOYMENT_YEARS",
    "PHONE_CHANGE_AGE; WORKING_CASH_EMPLOYMENT_YEARS",
    "PHONE_CHANGE_AGE; INCOME_TO_REGION_MEAN; INCOME_GAP_FROM_REGION_MEAN",
    "WORKING_CASH_EMPLOYMENT_YEARS",
    "INCOME_TO_REGION_MEAN; WORKING_CASH_EMPLOYMENT_YEARS",
    "extr2; PHONE_CHANGE_AGE; PHONE_CHANGE_ZERO; INCOME_TO_REGION_MEAN",
    "ID_PUBLISH_AGE; AGE_YEARS; INCOME_TO_REGION_MEAN; INCOME_GAP_FROM_REGION_MEAN; WORKING_CASH_INDICATOR; WORKING_CASH_EMPLOYMENT_YEARS",
    "REGISTRATION_AGE; INCOME_TO_REGION_MEAN; INCOME_GAP_FROM_REGION_MEAN; WORKING_CASH_INDICATOR; WORKING_CASH_EMPLOYMENT_YEARS",
    "extr2; REGISTRATION_AGE; PHONE_CHANGE_ZERO; INCOME_TO_REGION_MEAN; INCOME_GAP_FROM_REGION_MEAN",
    "extr2; ID_PUBLISH_AGE; PHONE_CHANGE_AGE; WORKING_CASH_INDICATOR; WORKING_CASH_EMPLOYMENT_YEARS",
    "PHONE_CHANGE_ZERO; INCOME_TO_REGION_MEAN; WORKING_CASH_INDICATOR; WORKING_CASH_EMPLOYMENT_YEARS",
    "ID_PUBLISH_AGE; AGE_YEARS; WORKING_CASH_INDICATOR; WORKING_CASH_EMPLOYMENT_YEARS",
    "PHONE_CHANGE_AGE; INCOME_TO_REGION_MEAN; WORKING_CASH_INDICATOR; WORKING_CASH_EMPLOYMENT_YEARS",
]

# CREDIT_TERM is removed from each row because it remains a fixed feature.
GROUP_2 = [
    "extr2; REGISTRATION_AGE; PHONE_CHANGE_ZERO",
    "extr2; PHONE_CHANGE_AGE; PHONE_CHANGE_ZERO",
    "REGISTRATION_AGE",
    "ID_PUBLISH_AGE; AGE_YEARS",
    "REGISTRATION_AGE; PHONE_CHANGE_ZERO; AGE_YEARS",
    "extr2; PHONE_CHANGE_ZERO; AGE_YEARS",
    "extr2; ID_PUBLISH_AGE; PHONE_CHANGE_AGE; PHONE_CHANGE_ZERO; AGE_YEARS",
    "REGISTRATION_AGE; PHONE_CHANGE_AGE; PHONE_CHANGE_ZERO; AGE_YEARS",
    "extr2; AGE_YEARS",
    "ID_PUBLISH_AGE; PHONE_CHANGE_AGE",
]

GROUP_3 = [
    "extr2; REGISTRATION_AGE; PHONE_CHANGE_ZERO",
    "extr2; PHONE_CHANGE_AGE; PHONE_CHANGE_ZERO",
    "ID_PUBLISH_AGE; PHONE_CHANGE_AGE; PHONE_CHANGE_ZERO; AGE_YEARS; INCOME_TYPE_TOP4_GROUP",
    "ID_PUBLISH_AGE; AGE_YEARS",
    "extr2; ID_PUBLISH_AGE; PHONE_CHANGE_AGE; PHONE_CHANGE_ZERO; AGE_YEARS",
    "REGISTRATION_AGE; PHONE_CHANGE_ZERO; OCCUPATION_TOP3_GROUP",
    "extr2; ID_PUBLISH_AGE; PHONE_CHANGE_AGE",
    "extr2; ID_PUBLISH_AGE; REGISTRATION_AGE; PHONE_CHANGE_ZERO",
    "extr2; PHONE_CHANGE_AGE",
    "AGE_YEARS; OCCUPATION_TOP3_GROUP",
    "REGISTRATION_AGE; PHONE_CHANGE_ZERO; AGE_YEARS",
    "ID_PUBLISH_AGE; REGISTRATION_AGE; PHONE_CHANGE_AGE; PHONE_CHANGE_ZERO; AGE_YEARS",
    "ID_PUBLISH_AGE; PHONE_CHANGE_ZERO; AGE_YEARS; OCCUPATION_TOP3_GROUP; INCOME_TYPE_TOP4_GROUP",
    "extr2; INCOME_TYPE_TOP4_GROUP",
    "PHONE_CHANGE_AGE; INCOME_TYPE_TOP4_GROUP",
]


def canonical_combo(features: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    clean = {str(x).strip() for x in features if str(x).strip()}
    unknown = clean - set(ALL_OPTIONAL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown optional feature(s): {sorted(unknown)}")
    return tuple(sorted(clean, key=lambda x: FEATURE_ORDER[x]))


def parse_group_combo(text: str) -> tuple[str, ...]:
    # Also tolerates an accidental fixed CREDIT_TERM in the source list.
    parts = [x.strip() for x in str(text).split(";") if x.strip()]
    parts = [x for x in parts if x != "CREDIT_TERM"]
    return canonical_combo(parts)


def combo_key(features: tuple[str, ...]) -> str:
    return "|".join(sorted(features))


def combo_label(features: tuple[str, ...]) -> str:
    return "; ".join(canonical_combo(features)) if features else "NONE"


def build_candidate_plan() -> list[dict[str, object]]:
    """Merge all three groups; keep source membership and first-seen order."""
    merged: dict[str, dict[str, object]] = {}
    for group_name, rows in (("GROUP_1", GROUP_1), ("GROUP_2", GROUP_2), ("GROUP_3", GROUP_3)):
        for rank, raw_combo in enumerate(rows, start=1):
            combo = parse_group_combo(raw_combo)
            key = combo_key(combo)
            if key not in merged:
                merged[key] = {
                    "features": combo,
                    "source_groups": [],
                    "source_ranks": [],
                }
            merged[key]["source_groups"].append(group_name)
            merged[key]["source_ranks"].append(f"{group_name}:{rank}")
    plan = []
    for number, item in enumerate(merged.values(), start=1):
        plan.append({
            "combination_id": f"C{number:02d}",
            "features": item["features"],
            "source_groups": "; ".join(item["source_groups"]),
            "source_ranks": "; ".join(item["source_ranks"]),
        })
    return plan


# ============================================================================
# 4. DATA / FEATURE ENGINEERING UTILITIES
# ============================================================================
def select_top_occupations(
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    min_count: int = MIN_CATEGORY_COUNT,
) -> tuple[list[str], pd.DataFrame]:
    mask = X_reference["AMT_GOODS_PRICE"].eq(SUBGROUP_GOODS_PRICE)
    reference = pd.DataFrame({
        "OCCUPATION": X_reference.loc[mask, "OCCUPATION_TYPE"].astype(str).to_numpy(),
        "TARGET": y_reference.loc[mask].to_numpy(),
    })
    reference = reference[reference["OCCUPATION"] != MISSING_TOKEN].copy()
    if reference.empty:
        return [], pd.DataFrame(columns=["APPLICANTS", "DEFAULTS", "DEFAULT_RATE"])
    stats = reference.groupby("OCCUPATION").agg(
        APPLICANTS=("TARGET", "size"),
        DEFAULTS=("TARGET", "sum"),
    )
    stats["DEFAULT_RATE"] = stats["DEFAULTS"] / stats["APPLICANTS"]
    eligible = stats[stats["APPLICANTS"] >= min_count].copy()
    eligible = eligible.sort_values(
        ["DEFAULT_RATE", "APPLICANTS"], ascending=[False, False]
    )
    return eligible.head(3).index.astype(str).tolist(), eligible


def fit_region_income_reference(
    X_reference: pd.DataFrame,
) -> tuple[pd.Series, float]:
    income = pd.to_numeric(X_reference["AMT_INCOME_TOTAL"], errors="coerce")
    region = pd.to_numeric(X_reference["REGION_RATING_CLIENT"], errors="coerce")
    reference = pd.DataFrame({
        "REGION_RATING_CLIENT": region,
        "AMT_INCOME_TOTAL": income,
    })
    global_mean = float(reference["AMT_INCOME_TOTAL"].mean())
    if not np.isfinite(global_mean):
        raise ValueError("Could not calculate the training-fold global income mean.")
    regional_means = (
        reference.dropna(subset=["REGION_RATING_CLIENT", "AMT_INCOME_TOTAL"])
        .groupby("REGION_RATING_CLIENT")["AMT_INCOME_TOTAL"]
        .mean()
    )
    return regional_means, global_mean


def make_engineered_features(
    df: pd.DataFrame,
    top_occupation_categories: list[str],
    regional_income_means: pd.Series,
    global_income_mean: float,
) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    credit = pd.to_numeric(df["AMT_CREDIT"], errors="coerce")
    annuity = pd.to_numeric(df["AMT_ANNUITY"], errors="coerce")
    features["CREDIT_TERM"] = credit / annuity.replace(0, np.nan)

    ext_sum = (
        pd.to_numeric(df["EXT_SOURCE_1"], errors="coerce")
        + pd.to_numeric(df["EXT_SOURCE_2"], errors="coerce")
        + pd.to_numeric(df["EXT_SOURCE_3"], errors="coerce")
        + pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce")
    )
    features["extr2"] = features["CREDIT_TERM"] * ext_sum
    features["ID_PUBLISH_AGE"] = -pd.to_numeric(df["DAYS_ID_PUBLISH"], errors="coerce")
    features["REGISTRATION_AGE"] = -pd.to_numeric(df["DAYS_REGISTRATION"], errors="coerce")

    phone = pd.to_numeric(df["DAYS_LAST_PHONE_CHANGE"], errors="coerce")
    features["PHONE_CHANGE_ZERO"] = phone.eq(0).astype(float).where(phone.notna(), np.nan)
    features["PHONE_CHANGE_AGE"] = -phone.mask(phone.eq(0))
    features["AGE_YEARS"] = -pd.to_numeric(df["DAYS_BIRTH"], errors="coerce") / 365.25

    income_type = df["NAME_INCOME_TYPE"].astype(str)
    in_subgroup = pd.to_numeric(df["AMT_GOODS_PRICE"], errors="coerce").eq(SUBGROUP_GOODS_PRICE)
    features["INCOME_TYPE_TOP4_GROUP"] = np.select(
        [~in_subgroup, income_type.eq(MISSING_TOKEN), income_type.isin(TOP_INCOME_TYPES)],
        ["OUTSIDE_450K_SUBGROUP", MISSING_TOKEN, "TOP4_INCOME_TYPES"],
        default="OTHER_INCOME_TYPES",
    )

    occupation = df["OCCUPATION_TYPE"].astype(str)
    features["OCCUPATION_TOP3_GROUP"] = np.select(
        [~in_subgroup, occupation.eq(MISSING_TOKEN), occupation.isin(top_occupation_categories)],
        ["OUTSIDE_450K_SUBGROUP", MISSING_TOKEN, "TOP3_HIGH_DEFAULT_OCCUPATIONS"],
        default="OTHER_OCCUPATIONS",
    )

    applicant_income = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce")
    region = pd.to_numeric(df["REGION_RATING_CLIENT"], errors="coerce")
    regional_mean = region.map(regional_income_means).fillna(global_income_mean)
    features["INCOME_TO_REGION_MEAN"] = applicant_income / regional_mean.replace(0, np.nan)
    features["INCOME_GAP_FROM_REGION_MEAN"] = applicant_income - regional_mean

    contract_type = df["NAME_CONTRACT_TYPE"].astype(str)
    working_cash_mask = (
        in_subgroup
        & income_type.eq("Working")
        & contract_type.eq("Cash loans")
    )
    features["WORKING_CASH_INDICATOR"] = working_cash_mask.astype(float)

    days_employed = pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce")
    employment_years = (-days_employed / 365.25).where(days_employed.lt(0))
    working_cash_years = pd.Series(0.0, index=df.index, dtype=float)
    working_cash_years.loc[working_cash_mask] = employment_years.loc[working_cash_mask]
    features["WORKING_CASH_EMPLOYMENT_YEARS"] = working_cash_years

    numeric_cols = features.select_dtypes(include=["number"]).columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return features


def prepare_raw_frames(train: pd.DataFrame, test: pd.DataFrame):
    if "TARGET" not in train or "SK_ID_CURR" not in train or "SK_ID_CURR" not in test:
        raise ValueError("Training requires TARGET and SK_ID_CURR; test requires SK_ID_CURR.")
    y = pd.to_numeric(train["TARGET"], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Expected binary TARGET values 0/1; got {sorted(y.unique())}")

    sample_ids = test["SK_ID_CURR"].copy()
    train_ids = train["SK_ID_CURR"].copy()
    X_base = train.drop(columns=["TARGET", "SK_ID_CURR"]).copy()
    X_test_base = test.drop(columns=["SK_ID_CURR"]).copy()

    missing_test = set(X_base.columns) - set(X_test_base.columns)
    extra_test = set(X_test_base.columns) - set(X_base.columns)
    if missing_test or extra_test:
        raise ValueError(f"Raw train/test schema mismatch. Missing={missing_test}; extra={extra_test}")
    X_test_base = X_test_base.reindex(columns=X_base.columns).copy()
    base_columns = X_base.columns.tolist()
    if len(base_columns) != 32:
        raise ValueError(f"Expected 32 raw features; found {len(base_columns)}")

    cat_features = X_base.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_features:
        X_base[col] = X_base[col].astype("object").where(X_base[col].notna(), MISSING_TOKEN).astype(str)
        X_test_base[col] = X_test_base[col].astype("object").where(X_test_base[col].notna(), MISSING_TOKEN).astype(str)
    return y, train_ids, sample_ids, X_base, X_test_base, base_columns, cat_features


def build_fold_matrices(
    X_base: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    base_columns: list[str],
):
    Xtr_raw = X_base.iloc[train_idx]
    Xva_raw = X_base.iloc[valid_idx]
    ytr = y.iloc[train_idx]

    # Both target-informed occupation selection and region means are fold-local.
    top_occupations, _ = select_top_occupations(Xtr_raw, ytr)
    regional_means, global_mean = fit_region_income_reference(Xtr_raw)

    eng_tr = make_engineered_features(Xtr_raw, top_occupations, regional_means, global_mean)
    eng_va = make_engineered_features(Xva_raw, top_occupations, regional_means, global_mean)

    if eng_tr.columns.tolist() != eng_va.columns.tolist():
        raise RuntimeError("Fold train/validation engineered schemas do not match.")
    if not eng_tr.columns.is_unique:
        raise RuntimeError("Duplicate engineered feature names.")
    if not set(FIXED_ENGINEERED_FEATURES + ALL_OPTIONAL_FEATURES).issubset(eng_tr.columns):
        raise RuntimeError("One or more configured engineered features were not generated.")

    Xtr_all = pd.concat([Xtr_raw, eng_tr], axis=1)
    Xva_all = pd.concat([Xva_raw, eng_va], axis=1)
    if Xtr_raw.columns.tolist() != base_columns:
        raise RuntimeError("Raw feature order changed in a fold.")
    return Xtr_all, Xva_all, top_occupations, float(global_mean)


# ============================================================================
# 5. CHECKPOINT / REPORT UTILITIES
# ============================================================================
def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_seed(base_seed: int, fold: int) -> int:
    # Same seed for every candidate within a fold, making model comparisons paired.
    return int((base_seed + fold * 1009) % 2_000_000_000)


def atomic_save_npz(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(tmp, path)


def atomic_save_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    yt = np.asarray(y_true, dtype=int)
    return {
        "roc_auc": float(roc_auc_score(yt, p)),
        "average_precision": float(average_precision_score(yt, p)),
        "logloss": float(log_loss(yt, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(yt, p)),
        "default_rate": float(np.mean(yt)),
    }


def make_run_signature(base_columns: list[str], candidate_plan: list[dict[str, object]]) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "raw_columns": base_columns,
        "candidate_plan": [
            {"key": combo_key(item["features"]), "sources": item["source_groups"]}
            for item in candidate_plan
        ],
        "cv": {"n_splits": N_SPLITS, "random_seed": RANDOM_SEED, "shuffle": True},
        "model": {
            "task_type": TASK_TYPE,
            "iterations": MAX_ITERATIONS,
            "learning_rate": LEARNING_RATE,
            "depth": DEPTH,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "l2_leaf_reg": L2_LEAF_REG,
        },
        "subgroup": SUBGROUP_GOODS_PRICE,
        "min_category_count": MIN_CATEGORY_COUNT,
        "top_income_types": TOP_INCOME_TYPES,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def checkpoint_file(combo: tuple[str, ...], fold: int) -> Path:
    key = combo_key(combo) if combo else "BASELINE"
    return CHECKPOINT_DIR / f"{sha1_text(key)[:16]}_fold{fold}.npz"


def load_checkpoint(
    path: Path,
    run_signature: str,
    combo: tuple[str, ...],
    fold: int,
    expected_valid_idx: np.ndarray,
):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            valid = (
                str(z["run_signature"].item()) == run_signature
                and str(z["combo_key"].item()) == combo_key(combo)
                and int(z["fold"].item()) == fold
                and np.array_equal(z["valid_indices"], expected_valid_idx)
            )
            if not valid:
                print(f"  Checkpoint mismatch; recomputing: {path.name}")
                return None
            probabilities = z["probabilities"].astype(float)
            best_iteration = int(z["best_iteration"].item())
            metrics = json.loads(str(z["metrics_json"].item()))
            runtime_seconds = float(z["runtime_seconds"].item())
            top_occupations = json.loads(str(z["top_occupations_json"].item()))
            global_income_mean = float(z["global_income_mean"].item())
        if len(probabilities) != len(expected_valid_idx) or not np.isfinite(probabilities).all():
            return None
        return probabilities, best_iteration, metrics, runtime_seconds, top_occupations, global_income_mean
    except Exception as exc:
        print(f"  Could not restore checkpoint {path.name}: {exc}; recomputing.")
        return None


def save_checkpoint(
    path: Path,
    run_signature: str,
    combo: tuple[str, ...],
    fold: int,
    valid_idx: np.ndarray,
    probabilities: np.ndarray,
    best_iteration: int,
    metrics: dict[str, float],
    runtime_seconds: float,
    top_occupations: list[str],
    global_income_mean: float,
) -> None:
    atomic_save_npz(
        path,
        run_signature=np.asarray(run_signature),
        combo_key=np.asarray(combo_key(combo)),
        fold=np.asarray(fold),
        valid_indices=np.asarray(valid_idx, dtype=np.int64),
        probabilities=np.asarray(probabilities, dtype=float),
        best_iteration=np.asarray(best_iteration),
        metrics_json=np.asarray(json.dumps(metrics, allow_nan=True)),
        runtime_seconds=np.asarray(runtime_seconds),
        top_occupations_json=np.asarray(json.dumps(top_occupations)),
        global_income_mean=np.asarray(global_income_mean),
    )


# ============================================================================
# 6. CATBOOST CONSTRUCTION
# ============================================================================
def make_model(seed: int) -> CatBoostClassifier:
    params = {
        "iterations": MAX_ITERATIONS,
        "learning_rate": LEARNING_RATE,
        "depth": DEPTH,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": seed,
        "allow_writing_files": False,
        "task_type": TASK_TYPE,
        "l2_leaf_reg": L2_LEAF_REG,
        "verbose": LOG_PERIOD,
    }
    if TASK_TYPE.upper() == "GPU":
        params["devices"] = "0"
    elif TASK_TYPE.upper() == "CPU":
        params["thread_count"] = -1
    else:
        raise ValueError("TASK_TYPE must be GPU or CPU.")
    return CatBoostClassifier(**params)


# ============================================================================
# 7. CV SUMMARY
# ============================================================================
def build_summary(
    candidate_plan: list[dict[str, object]],
    fold_records: dict[tuple[str, int], dict[str, object]],
    oof_by_key: dict[str, np.ndarray],
    y: pd.Series,
    n_rows: int,
) -> pd.DataFrame:
    baseline_key = ""
    baseline_fold_auc = {}
    baseline_fold_ap = {}
    for fold in range(1, N_SPLITS + 1):
        rec = fold_records.get((baseline_key, fold))
        if rec is not None:
            baseline_fold_auc[fold] = float(rec["roc_auc"])
            baseline_fold_ap[fold] = float(rec["average_precision"])

    entries = [{
        "combination_id": "BASELINE",
        "source_groups": "REFERENCE",
        "source_ranks": "REFERENCE",
        "optional_features": "NONE",
        "optional_feature_count": 0,
        "total_feature_count": len(RAW_BASE_COLUMNS) + 1,
        "is_baseline": True,
        "combo": tuple(),
    }]
    for item in candidate_plan:
        features = item["features"]
        entries.append({
            "combination_id": item["combination_id"],
            "source_groups": item["source_groups"],
            "source_ranks": item["source_ranks"],
            "optional_features": combo_label(features),
            "optional_feature_count": len(features),
            "total_feature_count": len(RAW_BASE_COLUMNS) + 1 + len(features),
            "is_baseline": False,
            "combo": features,
        })

    rows = []
    for entry in entries:
        key = combo_key(entry["combo"])
        records = [fold_records.get((key, fold)) for fold in range(1, N_SPLITS + 1)]
        records = [r for r in records if r is not None]
        if not records:
            continue
        aucs = np.asarray([float(r["roc_auc"]) for r in records])
        aps = np.asarray([float(r["average_precision"]) for r in records])
        loglosses = np.asarray([float(r["logloss"]) for r in records])
        briers = np.asarray([float(r["brier_score"]) for r in records])
        iters = np.asarray([int(r["best_iteration_zero_based"]) for r in records])

        oof = oof_by_key.get(key)
        if oof is not None and np.isfinite(oof).all():
            pooled = compute_metrics(y.to_numpy(), oof)
        else:
            pooled = {"roc_auc": np.nan, "average_precision": np.nan, "logloss": np.nan, "brier_score": np.nan, "default_rate": np.nan}

        paired_auc_deltas = [
            float(r["roc_auc"]) - baseline_fold_auc[int(r["fold"])]
            for r in records if int(r["fold"]) in baseline_fold_auc
        ]
        paired_ap_deltas = [
            float(r["average_precision"]) - baseline_fold_ap[int(r["fold"])]
            for r in records if int(r["fold"]) in baseline_fold_ap
        ]

        rows.append({
            "combination_id": entry["combination_id"],
            "source_groups": entry["source_groups"],
            "source_ranks": entry["source_ranks"],
            "optional_features": entry["optional_features"],
            "optional_feature_count": entry["optional_feature_count"],
            "total_feature_count": entry["total_feature_count"],
            "is_baseline": entry["is_baseline"],
            "completed_folds": len(records),
            "mean_fold_auc": float(np.mean(aucs)),
            "std_fold_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else np.nan,
            "min_fold_auc": float(np.min(aucs)),
            "max_fold_auc": float(np.max(aucs)),
            "mean_fold_average_precision": float(np.mean(aps)),
            "std_fold_average_precision": float(np.std(aps, ddof=1)) if len(aps) > 1 else np.nan,
            "mean_fold_logloss": float(np.mean(loglosses)),
            "std_fold_logloss": float(np.std(loglosses, ddof=1)) if len(loglosses) > 1 else np.nan,
            "mean_fold_brier": float(np.mean(briers)),
            "pooled_oof_auc": pooled["roc_auc"],
            "pooled_oof_average_precision": pooled["average_precision"],
            "pooled_oof_logloss": pooled["logloss"],
            "pooled_oof_brier": pooled["brier_score"],
            "median_best_iteration_zero_based": int(np.median(iters)),
            "refit_iterations": int(np.median(iters)) + 1,
            "positive_auc_folds_vs_baseline": int(np.sum(np.asarray(paired_auc_deltas) > 0)),
            "positive_ap_folds_vs_baseline": int(np.sum(np.asarray(paired_ap_deltas) > 0)),
            "mean_auc_delta_vs_baseline": float(np.mean(paired_auc_deltas)) if paired_auc_deltas else np.nan,
            "mean_ap_delta_vs_baseline": float(np.mean(paired_ap_deltas)) if paired_ap_deltas else np.nan,
            "mean_logloss_delta_vs_baseline": float(np.mean(loglosses) - np.mean([float(r["logloss"]) for r in fold_records.values() if r.get("combo_key") == "" ])) if entry["is_baseline"] is False else 0.0,
            "mean_runtime_seconds": float(np.mean([float(r["runtime_seconds"]) for r in records])),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    # Prefer mean fold AUC; then AP; then lower logloss; then lower AUC variability.
    summary = summary.sort_values(
        ["mean_fold_auc", "mean_fold_average_precision", "mean_fold_logloss", "std_fold_auc"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


# ============================================================================
# 8. MAIN
# ============================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    train = pd.read_csv(TRAIN_FILE, low_memory=False)
    test = pd.read_csv(TEST_FILE, low_memory=False)
    print("Train shape:", train.shape, flush=True)
    print("Test shape:", test.shape, flush=True)

    y, train_ids, sample_ids, X_base, X_test_base, base_columns, cat_features = prepare_raw_frames(train, test)
    global RAW_BASE_COLUMNS
    RAW_BASE_COLUMNS = base_columns

    print("\nRaw features retained:", len(base_columns), flush=True)
    print("Raw categorical features:", cat_features, flush=True)
    print("Fixed engineered features:", FIXED_ENGINEERED_FEATURES, flush=True)
    print("Optional feature pool:", len(ALL_OPTIONAL_FEATURES), flush=True)

    # Validate the plan count and duplicate accounting before any training.
    if (len(GROUP_1), len(GROUP_2), len(GROUP_3)) != (15, 10, 15):
        raise RuntimeError("Candidate group definitions do not match the supplied 15/10/15 lists.")
    candidate_plan = build_candidate_plan()
    if len(candidate_plan) != 35:
        raise RuntimeError(f"Expected 35 unique candidate combinations, found {len(candidate_plan)}")

    group1_keys = {combo_key(parse_group_combo(x)) for x in GROUP_1}
    group2_keys = {combo_key(parse_group_combo(x)) for x in GROUP_2}
    group3_keys = {combo_key(parse_group_combo(x)) for x in GROUP_3}
    print("\nCandidate consolidation:", flush=True)
    print("Group 1 rows / unique:", len(GROUP_1), len(group1_keys), flush=True)
    print("Group 2 rows / unique:", len(GROUP_2), len(group2_keys), flush=True)
    print("Group 3 rows / unique:", len(GROUP_3), len(group3_keys), flush=True)
    print("Group 2/3 overlap removed:", len(group2_keys & group3_keys), flush=True)
    print("Unique candidate combinations:", len(candidate_plan), flush=True)
    print("Reference baseline:", 1, flush=True)
    print("Total combinations to evaluate:", len(candidate_plan) + 1, flush=True)
    print("Total CV fits:", (len(candidate_plan) + 1) * N_SPLITS, flush=True)

    run_signature = make_run_signature(base_columns, candidate_plan)
    print("Run signature:", run_signature[:16], flush=True)

    # One identical set of folds is reused for all candidate combinations.
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    splits = [(tr.copy(), va.copy()) for tr, va in cv.split(np.zeros(len(y)), y.to_numpy())]

    # Mapping of key -> candidate information. Include baseline first.
    all_runs = [{
        "combination_id": "BASELINE",
        "features": tuple(),
        "source_groups": "REFERENCE",
        "source_ranks": "REFERENCE",
    }] + candidate_plan

    fold_records: dict[tuple[str, int], dict[str, object]] = {}
    oof_by_key: dict[str, np.ndarray] = {
        combo_key(item["features"]): np.full(len(y), np.nan, dtype=float)
        for item in all_runs
    }

    model_params_for_print = {
        "task_type": TASK_TYPE,
        "iterations": MAX_ITERATIONS,
        "learning_rate": LEARNING_RATE,
        "depth": DEPTH,
        "l2_leaf_reg": L2_LEAF_REG,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }
    print("\nCatBoost configuration:", model_params_for_print, flush=True)

    total_fits = len(all_runs) * N_SPLITS
    fit_counter = 0

    for run in all_runs:
        combo = run["features"]
        key = combo_key(combo)
        combination_id = run["combination_id"]
        feature_columns = base_columns + FIXED_ENGINEERED_FEATURES + list(combo)
        run_cat_features = cat_features.copy() + [
            f for f in combo if f in CATEGORICAL_ENGINEERED_FEATURES
        ]

        print("\n" + "#" * 84, flush=True)
        print(f"COMBINATION {combination_id}: {combo_label(combo)}", flush=True)
        print("Sources:", run["source_groups"], flush=True)
        print("Raw features retained:", len(base_columns), flush=True)
        print("Fixed features:", FIXED_ENGINEERED_FEATURES, flush=True)
        print("Total feature count:", len(feature_columns), flush=True)
        print("#" * 84, flush=True)

        for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
            fit_counter += 1
            cp_path = checkpoint_file(combo, fold_number)
            restored = load_checkpoint(cp_path, run_signature, combo, fold_number, valid_idx)

            if restored is not None:
                probabilities, best_iteration, metrics, runtime_seconds, top_occupations, global_income_mean = restored
                print(
                    f"[{fit_counter}/{total_fits}] {combination_id} Fold {fold_number}/{N_SPLITS}: "
                    f"restored checkpoint | AUC={metrics['roc_auc']:.6f}",
                    flush=True,
                )
            else:
                fold_started = perf_counter()
                Xtr_all, Xva_all, top_occupations, global_income_mean = build_fold_matrices(
                    X_base, y, train_idx, valid_idx, base_columns
                )
                X_train_run = Xtr_all[feature_columns]
                X_valid_run = Xva_all[feature_columns]
                y_train_fold = y.iloc[train_idx]
                y_valid_fold = y.iloc[valid_idx]

                model = None
                try:
                    model = make_model(stable_seed(RANDOM_SEED, fold_number))
                    model.fit(
                        X_train_run,
                        y_train_fold,
                        cat_features=run_cat_features,
                        eval_set=(X_valid_run, y_valid_fold),
                        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        use_best_model=True,
                        verbose=LOG_PERIOD,
                    )
                    classes = list(model.classes_)
                    if 1 not in classes:
                        raise RuntimeError(f"Positive class 1 missing from model classes: {classes}")
                    positive_index = classes.index(1)
                    probabilities = model.predict_proba(X_valid_run)[:, positive_index]
                    best_iteration = int(model.get_best_iteration())
                    if best_iteration < 0:
                        best_iteration = int(model.tree_count_ - 1)
                    metrics = compute_metrics(y_valid_fold.to_numpy(), probabilities)
                    runtime_seconds = perf_counter() - fold_started
                    save_checkpoint(
                        cp_path, run_signature, combo, fold_number, valid_idx,
                        probabilities, best_iteration, metrics, runtime_seconds,
                        top_occupations, global_income_mean,
                    )
                    print(
                        f"[{fit_counter}/{total_fits}] {combination_id} Fold {fold_number}/{N_SPLITS}: "
                        f"AUC={metrics['roc_auc']:.6f} | AP={metrics['average_precision']:.6f} | "
                        f"Logloss={metrics['logloss']:.6f} | best_iter={best_iteration} | "
                        f"runtime={runtime_seconds:.1f}s",
                        flush=True,
                    )
                finally:
                    if model is not None:
                        del model
                    gc.collect()

            oof_by_key[key][valid_idx] = probabilities
            record = {
                "combination_id": combination_id,
                "combo_key": key,
                "source_groups": run["source_groups"],
                "source_ranks": run["source_ranks"],
                "fold": fold_number,
                "optional_features": combo_label(combo),
                "optional_feature_count": len(combo),
                "total_feature_count": len(feature_columns),
                "roc_auc": float(metrics["roc_auc"]),
                "average_precision": float(metrics["average_precision"]),
                "logloss": float(metrics["logloss"]),
                "brier_score": float(metrics["brier_score"]),
                "default_rate": float(metrics["default_rate"]),
                "best_iteration_zero_based": int(best_iteration),
                "iterations_for_refit": int(best_iteration) + 1,
                "runtime_seconds": round(float(runtime_seconds), 2),
                "top3_occupations_fold_train": json.dumps(top_occupations),
                "region_income_global_mean_fold_train": float(global_income_mean),
                "status": "OK",
            }
            fold_records[(key, fold_number)] = record

            # Persist fold report after every completed/restored fold.
            fold_df = pd.DataFrame(list(fold_records.values()))
            fold_df = fold_df.sort_values(["combination_id", "fold"])
            atomic_save_csv(fold_df, FOLD_REPORT_FILE)

        # Confirm all five validation folds contributed to the OOF vector.
        if not np.isfinite(oof_by_key[key]).all():
            raise RuntimeError(f"OOF predictions are incomplete for {combination_id}.")
        pooled = compute_metrics(y.to_numpy(), oof_by_key[key])
        print(
            f"COMPLETED {combination_id}: pooled OOF AUC={pooled['roc_auc']:.6f}, "
            f"AP={pooled['average_precision']:.6f}, Logloss={pooled['logloss']:.6f}",
            flush=True,
        )

        # Save the current OOF matrix incrementally so a later interruption
        # does not lose already completed combination predictions.
        current_oof = pd.DataFrame({
            "SK_ID_CURR": train_ids.to_numpy(),
            "TARGET": y.to_numpy(),
        })
        for prior_run in all_runs:
            prior_key = combo_key(prior_run["features"])
            pred = oof_by_key[prior_key]
            if np.isfinite(pred).all():
                col = prior_run["combination_id"]
                current_oof[f"P_DEFAULT_{col}"] = pred
        temp_oof = OOF_PREDICTIONS_FILE.with_suffix(OOF_PREDICTIONS_FILE.suffix + ".tmp")
        current_oof.to_csv(temp_oof, index=False, compression="gzip")
        os.replace(temp_oof, OOF_PREDICTIONS_FILE)

    # ========================================================================
    # 9. BUILD AND SAVE SUMMARY REPORT
    # ========================================================================
    summary = build_summary(candidate_plan, fold_records, oof_by_key, y, len(y))
    if summary.empty or summary["completed_folds"].min() != N_SPLITS:
        raise RuntimeError("At least one combination does not have all five completed folds.")
    atomic_save_csv(summary, SUMMARY_REPORT_FILE)

    print("\n" + "#" * 88, flush=True)
    print("STRATIFIED 5-FOLD COMBINATION SEARCH COMPLETE", flush=True)
    print("#" * 88, flush=True)
    print("Unique candidates from the three groups:", len(candidate_plan), flush=True)
    print("Reference baseline:", 1, flush=True)
    print("Total combinations:", len(all_runs), flush=True)
    print("Successful CV fits:", len(fold_records), "/", total_fits, flush=True)
    print("\nTOP 20 BY MEAN FOLD AUC:", flush=True)
    show_cols = [
        "rank", "combination_id", "source_groups", "optional_features",
        "mean_fold_auc", "std_fold_auc", "mean_fold_average_precision",
        "mean_fold_logloss", "positive_auc_folds_vs_baseline",
        "median_best_iteration_zero_based",
    ]
    print(summary[show_cols].head(20).to_string(index=False), flush=True)

    # ========================================================================
    # 10. SELECT BEST CV COMBINATION AND REFIT ON ALL TRAINING ROWS
    # ========================================================================
    best_row = summary.iloc[0]
    best_id = str(best_row["combination_id"])
    if best_id == "BASELINE":
        best_combo = tuple()
    else:
        best_combo = next(item["features"] for item in candidate_plan if item["combination_id"] == best_id)

    best_key = combo_key(best_combo)
    best_fold_iterations = [
        int(fold_records[(best_key, fold)]["best_iteration_zero_based"])
        for fold in range(1, N_SPLITS + 1)
    ]
    refit_iterations = max(1, int(np.median(best_fold_iterations)) + 1)

    print("\nSelected CV combination:", best_id, flush=True)
    print("Selected optional features:", combo_label(best_combo), flush=True)
    print("Mean Fold AUC:", f"{float(best_row['mean_fold_auc']):.6f}", flush=True)
    print("Pooled OOF AUC:", f"{float(best_row['pooled_oof_auc']):.6f}", flush=True)
    print("Fold best iterations:", best_fold_iterations, flush=True)
    print("Full-data refit iterations (median + 1):", refit_iterations, flush=True)

    # For the final fit, supervised occupation categories are learned on all
    # labeled training rows; regional means are also refit on all training rows.
    final_top_occupations, final_occupation_ranking = select_top_occupations(X_base, y)
    final_region_means, final_global_income_mean = fit_region_income_reference(X_base)
    final_eng_train = make_engineered_features(
        X_base, final_top_occupations, final_region_means, final_global_income_mean
    )
    final_eng_test = make_engineered_features(
        X_test_base, final_top_occupations, final_region_means, final_global_income_mean
    )
    X_all_final = pd.concat([X_base, final_eng_train], axis=1)
    X_test_all_final = pd.concat([X_test_base, final_eng_test], axis=1)
    if not X_all_final.columns.equals(X_test_all_final.columns):
        raise RuntimeError("Final training/test feature schemas do not match.")

    final_feature_columns = base_columns + FIXED_ENGINEERED_FEATURES + list(best_combo)
    final_cat_features = cat_features.copy() + [
        f for f in best_combo if f in CATEGORICAL_ENGINEERED_FEATURES
    ]

    final_params = {
        "iterations": refit_iterations,
        "learning_rate": LEARNING_RATE,
        "depth": DEPTH,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "task_type": TASK_TYPE,
        "l2_leaf_reg": L2_LEAF_REG,
        "verbose": LOG_PERIOD,
    }
    if TASK_TYPE.upper() == "GPU":
        final_params["devices"] = "0"
    else:
        final_params["thread_count"] = -1

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(
        X_all_final[final_feature_columns],
        y,
        cat_features=final_cat_features,
        verbose=LOG_PERIOD,
    )
    final_model.save_model(str(BEST_MODEL_FILE))

    # Feature importance for the selected final model.
    importance_values = final_model.get_feature_importance()
    importance = pd.DataFrame({
        "feature": final_feature_columns,
        "importance": importance_values,
    }).sort_values("importance", ascending=False)
    importance.to_csv(BEST_IMPORTANCE_FILE, index=False)

    final_classes = list(final_model.classes_)
    if 1 not in final_classes:
        raise RuntimeError(f"Positive class 1 absent from final model: {final_classes}")
    positive_index = final_classes.index(1)
    test_probabilities = final_model.predict_proba(
        X_test_all_final[final_feature_columns]
    )[:, positive_index]

    submission = pd.DataFrame({
        "SK_ID_CURR": sample_ids.to_numpy(),
        "TARGET": test_probabilities,
    })
    if (
        len(submission) != len(test)
        or not submission["SK_ID_CURR"].is_unique
        or submission["TARGET"].isna().any()
        or not np.isfinite(submission["TARGET"]).all()
        or not submission["TARGET"].between(0, 1).all()
    ):
        raise RuntimeError("Submission validation failed.")
    submission.to_csv(BEST_SUBMISSION_FILE, index=False)

    metadata = {
        "script_version": SCRIPT_VERSION,
        "cv_method": "StratifiedKFold",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_seed": RANDOM_SEED,
        "candidate_groups": {
            "group_1_submitted": len(GROUP_1),
            "group_2_submitted": len(GROUP_2),
            "group_3_submitted": len(GROUP_3),
            "group_2_group_3_duplicates_removed": len(group2_keys & group3_keys),
            "unique_candidates": len(candidate_plan),
            "baseline_included": True,
            "total_combinations_evaluated": len(all_runs),
            "total_cv_fits": total_fits,
        },
        "model_parameters": model_params_for_print,
        "raw_features_preserved": True,
        "raw_feature_count": len(base_columns),
        "raw_features": base_columns,
        "fixed_engineered_features": FIXED_ENGINEERED_FEATURES,
        "selected_optional_features": list(best_combo),
        "selected_combination_id": best_id,
        "selected_source_groups": str(best_row["source_groups"]),
        "selected_mean_fold_auc": float(best_row["mean_fold_auc"]),
        "selected_std_fold_auc": float(best_row["std_fold_auc"]),
        "selected_pooled_oof_auc": float(best_row["pooled_oof_auc"]),
        "selected_mean_fold_average_precision": float(best_row["mean_fold_average_precision"]),
        "selected_mean_fold_logloss": float(best_row["mean_fold_logloss"]),
        "fold_best_iterations_zero_based": best_fold_iterations,
        "refit_iterations": refit_iterations,
        "categorical_features": final_cat_features,
        "final_top3_occupations": final_top_occupations,
        "final_top3_occupation_ranking_top10": final_occupation_ranking.head(10).reset_index().to_dict(orient="records"),
        "top4_income_types": TOP_INCOME_TYPES,
        "region_income_reference_fit_on": "all_labeled_training_rows_for_final_refit",
        "region_income_global_mean": float(final_global_income_mean),
        "region_income_groups": int(len(final_region_means)),
        "fold_metrics_csv": str(FOLD_REPORT_FILE.resolve()),
        "summary_csv": str(SUMMARY_REPORT_FILE.resolve()),
        "oof_predictions_csv_gz": str(OOF_PREDICTIONS_FILE.resolve()),
        "model_file": str(BEST_MODEL_FILE.resolve()),
        "feature_importance_file": str(BEST_IMPORTANCE_FILE.resolve()),
        "submission_file": str(BEST_SUBMISSION_FILE.resolve()),
    }
    BEST_FEATURES_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "#" * 88, flush=True)
    print("FINAL ARTIFACTS", flush=True)
    print("#" * 88, flush=True)
    print("Fold metrics:", FOLD_REPORT_FILE.resolve(), flush=True)
    print("CV summary:", SUMMARY_REPORT_FILE.resolve(), flush=True)
    print("OOF predictions:", OOF_PREDICTIONS_FILE.resolve(), flush=True)
    print("Best model:", BEST_MODEL_FILE.resolve(), flush=True)
    print("Best feature metadata:", BEST_FEATURES_FILE.resolve(), flush=True)
    print("Feature importance:", BEST_IMPORTANCE_FILE.resolve(), flush=True)
    print("Submission:", BEST_SUBMISSION_FILE.resolve(), flush=True)
    print("Submission shape:", submission.shape, flush=True)
    print("\nNote: upload the submission to the competition to obtain its leaderboard score.", flush=True)


if __name__ == "__main__":
    main()
