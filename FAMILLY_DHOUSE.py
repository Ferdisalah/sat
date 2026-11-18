# -*- coding: utf-8 -*-
"""
HOME CREDIT — SIX-MODEL WINDOWS / NVIDIA GPU PIPELINE

Purpose
-------
Extend the existing four-model Home Credit pipeline with only two selected
features and two additional CatBoost tree-growth policies.

Inputs
------
  1. Downloads/train.csv
  2. Downloads/test (1).csv
  3. Previously saved fold-local EXT_SOURCE imputation artifacts under
     CATBOOST_FULLDATA_TWO_RATIO_FEATURES_EXT_SOURCE_IMPUTATION_GRID

The sample submission is optional and used only for ID/schema validation.

Reused artifacts
----------------
- Five exact StratifiedKFold(5, shuffle=True, random_state=42) splits.
- Five saved IMPUTED_VALUES.npz files (three EXT_SOURCE arrays each).
- Fifteen saved CatBoost imputer models (three per fold), used only to impute
  TEST rows with the matching fold model.
- Persistent fold-local GMM cache at Downloads/HOME_CREDIT_GMM_MODEL_CACHE.

Classifier features
-------------------
The original four-model feature pipeline is retained, including:
- imputed EXT_SOURCE_1/2/3 and original missingness flags;
- CREDIT_TERM and LOAN_STRUCTURE_CLUSTER;
- LOCAL_EXT_SOURCE_AGE_INTERACTION;
- original tree / RealMLP engineered features.

Only two selected features are added:
- DAYS_ID_TO_BIRTH = DAYS_ID_PUBLISH / DAYS_BIRTH (numeric)
- FAMILLY_DHOUSE: hierarchical categorical feature based on
  CODE_GENDER + NAME_FAMILY_STATUS + NAME_HOUSING_TYPE, with a minimum
  reference support of 40 rows and fold-training-only counts.

Six classifiers, five outer folds each
--------------------------------------
1. RealMLP (CUDA)
2. CatBoost SymmetricTree (CUDA)
3. LightGBM (CPU)
4. XGBoost (CUDA)
5. CatBoost Depthwise (CUDA)
6. CatBoost Lossguide (CUDA)

CatBoost iteration-selection protocol
--------------------------------------
For each outer fold, all three CatBoost growth policies use the same persisted
15% stratified inner-validation split. Each pilot model is trained with a
maximum of 4200 iterations, AUC evaluation, and 150-round early stopping.
The selected tree count is then used to refit a fresh model on the complete
outer-training fold. The outer validation fold is excluded from iteration
selection and is used only for final OOF evaluation.
This yields 15 CatBoost pilot fits plus 30 final classifier fits (45 fit calls
in total, of which 30 final models are archived).

The four-model equal-weight ensemble is retained as a baseline only. No
six-model equal-weight ensemble is created: all six OOF and fold-wise TEST
predictions are saved for the later advanced blending / model-removal stage.

Important
---------
- This script does NOT retrain EXT_SOURCE imputers.
- It validates that the saved imputation fold indices match the active folds.
- It does NOT add GOODS_PRICE_TO_ANNUITY to classifier features. That variable
  is reconstructed only when creating inputs for the saved imputers.
- GMM fitting and transformation remain target-free and fold-local.
- All fold-specific models, predictions, feature caches and preprocessing
  artifacts are archived independently.
"""
from __future__ import annotations

# Set CUDA visibility before importing torch / CUDA-dependent libraries.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
_BOOTSTRAP_CPU_THREADS = max(1, min(2, os.cpu_count() or 1))
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = str(_BOOTSTRAP_CPU_THREADS)

import gc
import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from pytabkit import RealMLP_TD_Classifier
from packaging.version import Version
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")


# =============================================================================
# 1. WINDOWS PATHS — THE ONLY SECTION THAT SHOULD NEED PATH EDITS
# =============================================================================

DATA_DIR = Path(r"C:\Users\UNITECH\Downloads")
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test (1).csv"
SAMPLE_SUB_FILE = DATA_DIR / "sample_submission.csv"  # optional

IMPUTATION_EXPERIMENT_DIR = (
    DATA_DIR / "CATBOOST_FULLDATA_TWO_RATIO_FEATURES_EXT_SOURCE_IMPUTATION_GRID"
)
IMPUTATION_MODELS_DIR = IMPUTATION_EXPERIMENT_DIR / "IMPUTATION_MODELS"
IMPUTATION_CLASSIFIER_SCENARIO_DIR = (
    IMPUTATION_EXPERIMENT_DIR / "SCENARIOS" / "IMPUTE_EXT_SOURCE_1_2_3"
)

# Persistent GMM cache used by the previous GMM workflow.
GMM_CACHE_DIR = DATA_DIR / "HOME_CREDIT_GMM_MODEL_CACHE"

# New output directory: never overwrite the earlier four-model run.
OUTPUT_DIR = (
    DATA_DIR
    / "outputwpw"
    / "HOME_CREDIT_SIX_MODEL_GPU_IMPUTED_EXT_SOURCE_1_2_3_PLUS_TWO_FEATURES_INNER_VALIDATION_REFIT_V2_FAMILLY_DHOUSE"
)

IMPUTATION_TEST_DIR = OUTPUT_DIR / "IMPUTATION_TEST_VALUES"
GMM_DIR = OUTPUT_DIR / "GMM_SEGMENTATION"
CACHE_DIR = OUTPUT_DIR / "FEATURE_CACHE"
PREPROCESSING_DIR = OUTPUT_DIR / "PREPROCESSING"
MODEL_DIR = OUTPUT_DIR / "CLASSIFICATION_MODELS"
PREDICTION_DIR = OUTPUT_DIR / "FOLD_PREDICTIONS"
IMPORTANCE_DIR = OUTPUT_DIR / "FEATURE_IMPORTANCE"
ENSEMBLE_DIR = OUTPUT_DIR / "ENSEMBLE"
CHECKPOINT_DIR = OUTPUT_DIR / "CHECKPOINTS"

for _directory in [
    OUTPUT_DIR,
    IMPUTATION_TEST_DIR,
    GMM_DIR,
    CACHE_DIR,
    PREPROCESSING_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    IMPORTANCE_DIR,
    ENSEMBLE_DIR,
    CHECKPOINT_DIR,
    GMM_CACHE_DIR,
]:
    _directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
RANDOM_SEED = 42
N_SPLITS = 5

EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
EXT_SOURCE_MISSING_COLUMNS = [f"{column}_MISSING" for column in EXT_SOURCE_COLUMNS]
MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY = "__UNKNOWN__"

LOAN_STRUCTURE_FEATURE = "LOAN_STRUCTURE_CLUSTER"
NEW_INTERACTION_FEATURE = "LOCAL_EXT_SOURCE_AGE_INTERACTION"
FAMILLY_DHOUSE_FEATURE = "FAMILLY_DHOUSE"
FAMILLY_DHOUSE_MIN_COUNT = 40


N_NEIGHBORS = 500
NEIGHBOR_QUERY_CHUNK_SIZE = 2048
CPU_THREADS = max(1, min(2, os.cpu_count() or 1))

# Saved EXT_SOURCE imputers in the supplied experiment have 35 input features.
EXPECTED_SAVED_IMPUTER_FEATURE_COUNT = 35

# Original four-model fixed parameters.
FIXED_CATBOOST_PARAMS = {
    "iterations": 4200,
    "learning_rate": 0.030207171090683906,
    "depth": 4,
    "l2_leaf_reg": 18.17422221879975,
    "random_strength": 0.6746151640000321,
    "border_count": 128,
}

FIXED_LIGHTGBM_PARAMS = {
    "max_depth": 8,
    "num_leaves": 30,
    "n_estimators": 3800,
    "learning_rate": 0.015831790730798656,
    "min_child_samples": 280,
    "reg_lambda": 0.15805460542063748,
    "reg_alpha": 3.920649538192273e-05,
    "colsample_bytree": 0.6054519676764284,
    "subsample": 0.9935009272600186,
    "min_split_gain": 0.9053872910470832,
    "max_bin": 1023,
    "cat_smooth": 61.33613152281461,
    "cat_l2": 2.653555349507821,
}

FIXED_XGBOOST_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03162931523007232,
    "max_depth": 4,
    "min_child_weight": 43.61361810092031,
    "gamma": 4.251923725207756,
    "subsample": 0.6324027744325311,
    "colsample_bytree": 0.5996292985062543,
    "reg_lambda": 0.2204066172541325,
    "reg_alpha": 7.878776514694756,
    "max_bin": 1024,
    "max_cat_to_onehot": 9,
}

# All three CatBoost growth policies use the same inner-validation tree-count
# selection procedure, then refit on the complete outer-training fold.
# Architecture-specific growth/leaf constraints remain unchanged.
ASYMMETRIC_MIN_DATA_IN_LEAF = 100
LOSSGUIDE_MAX_LEAVES = 32
CATBOOST_EARLY_STOPPING_ROUNDS = 200
INNER_VALIDATION_FRACTION = 0.15
INNER_VALIDATION_SEED_OFFSET = 10_000

REALMLP_PARAMS = {
    "device": "cuda:0",
    "random_state": RANDOM_SEED,
    "n_cv": 1,
    "n_refit": 0,
    "n_threads": CPU_THREADS,
    "n_epochs": 256,
    "batch_size": 512,
    "predict_batch_size": 2048,
    "hidden_sizes": [256, 256, 256],
    "lr": 0.03,
    "p_drop": 0.15,
    "wd": 0.01,
    "val_metric_name": "cross_entropy",
    "use_ls": False,
    "verbosity": 1,
}

MLP_RAW_CATEGORICAL_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]
MLP_CATEGORICAL_COLUMNS = (
    MLP_RAW_CATEGORICAL_COLUMNS
    + [LOAN_STRUCTURE_FEATURE, FAMILLY_DHOUSE_FEATURE]
)

MLP_ENGINEERED_FEATURES = [
    "CREDIT_TERM",
    "INCOME_TO_REGION_MEAN",
    "OWN_CAR_AGE_CLEAN",
    "DIFF_RATIO",
    NEW_INTERACTION_FEATURE,
  
    FAMILLY_DHOUSE_FEATURE,
    LOAN_STRUCTURE_FEATURE,
]
TREE_ENGINEERED_FEATURES = [
    "CREDIT_TERM",
    "INCOME_TO_REGION_MEAN",
    "OWN_CAR_AGE_CLEAN",
    "DIFF_RATIO",
    NEW_INTERACTION_FEATURE,
  
    FAMILLY_DHOUSE_FEATURE,
    LOAN_STRUCTURE_FEATURE,
]

BASELINE_MODEL_NAMES = ["realmlp", "catboost_symmetric", "lightgbm", "xgboost"]
MODEL_NAMES = [
    "realmlp",
    "catboost_symmetric",
    "lightgbm",
    "xgboost",
    "catboost_depthwise",
    "catboost_lossguide",
]
BASELINE_ENSEMBLE_WEIGHTS = {
    "realmlp": 0.25,
    "catboost_symmetric": 0.25,
    "lightgbm": 0.25,
    "xgboost": 0.25,
}

# GMM settings: deliberately kept compatible with the persistent GMM cache key.
LOAN_STRUCTURE_TERM_PRECISION = 2
LOAN_STRUCTURE_GOODS_PRECISION = 3
LOAN_STRUCTURE_SPIKE_MIN_SHARE = 0.03
LOAN_STRUCTURE_SPIKE_DOMINANCE = 2.0
GMM_MAX_COMPONENTS = 30
GMM_N_INIT = 3
GMM_MAX_ITER = 300
GMM_REG_COVAR = 1e-5
GMM_MIN_COMPONENT_ROWS = 100
GMM_RANDOM_STATE = RANDOM_SEED
GMM_STABILITY_REPEATS = 3
TERM_SPIKE_ONLY_REVIEW_SUPPORT = 500
GMM_CACHE_SCHEMA_VERSION = 1
GMM_CACHE_INPUT_COLUMNS = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]


# =============================================================================
# 3. ENVIRONMENT AND VALIDATION
# =============================================================================

def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def print_section(title: str) -> None:
    print("\n" + "=" * 105, flush=True)
    print(title, flush=True)
    print("=" * 105, flush=True)


def detect_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False


if not detect_nvidia_gpu():
    raise RuntimeError(
        "nvidia-smi did not detect an NVIDIA GPU. Check the Windows NVIDIA driver."
    )
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA-enabled PyTorch is required for RealMLP. "
        f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}"
    )
if Version(xgb.__version__.split("+")[0]) < Version("2.0.0"):
    raise RuntimeError(f"XGBoost >= 2.0.0 is required; installed={xgb.__version__}")

try:
    torch.cuda.set_device(0)
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(max(1, min(2, CPU_THREADS)))
except RuntimeError:
    pass

DEVICE = "cuda:0"
REALMLP_PARAMS["device"] = DEVICE

print_section("HOME CREDIT — SIX-MODEL GPU PIPELINE")
print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"PyTorch CUDA : {torch.version.cuda}")
print(f"NVIDIA GPU   : {torch.cuda.get_device_name(0)}")
print(f"GPU memory   : {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GiB")
print(f"CatBoost     : {package_version('catboost')} (GPU)")
print(f"LightGBM     : {package_version('lightgbm')} (CPU)")
print(f"XGBoost      : {xgb.__version__} (CUDA)")
print(f"PyTabKit     : {package_version('pytabkit')} (RealMLP CUDA)")
print(f"CPU threads  : {CPU_THREADS}")
print(f"Output       : {OUTPUT_DIR}")
print(f"Imputation   : {IMPUTATION_EXPERIMENT_DIR}")
print(f"GMM cache    : {GMM_CACHE_DIR}")


# =============================================================================
# 4. ATOMIC ARTIFACT / CHECKPOINT / METRIC HELPERS
# =============================================================================

def tmp_path(path: Path) -> Path:
    return path.with_name(path.stem + ".tmp" + path.suffix)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_path(path)
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_path(path)
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_path(path)
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_path(path)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_model_save(model: Any, path: Path, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_path(path)
    if kind == "catboost":
        model.save_model(str(temporary), format="cbm")
    elif kind == "lightgbm":
        model.booster_.save_model(str(temporary))
    elif kind == "xgboost":
        model.save_model(str(temporary))
    elif kind == "pickle":
        with temporary.open("wb") as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
    else:
        raise ValueError(f"Unsupported model kind: {kind}")
    os.replace(temporary, path)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def sha256_json(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def checkpoint_complete(
    marker_path: Path,
    expected_stage: str,
    config_hash: str,
    required_files: list[Path] | None = None,
) -> bool:
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("stage") != expected_stage:
            return False
        if marker.get("config_sha256") != config_hash:
            return False
        if required_files and not all(path.is_file() for path in required_files):
            return False
        for artifact in marker.get("artifacts", []):
            artifact_path = Path(artifact["path"])
            if not artifact_path.is_file():
                return False
            if int(artifact.get("size_bytes", -1)) != int(artifact_path.stat().st_size):
                return False
        return True
    except Exception:
        return False


def mark_complete(
    marker_path: Path,
    stage: str,
    config_hash: str,
    artifacts: list[Path],
) -> None:
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot mark {stage} complete; missing artifacts: {missing}")
    atomic_json(
        marker_path,
        {
            "stage": stage,
            "config_sha256": config_hash,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": [
                {"path": str(path), "size_bytes": int(path.stat().st_size)}
                for path in artifacts
            ],
        },
    )


def check_probabilities(name: str, probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{name}: expected finite one-dimensional probabilities")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{name}: probabilities outside [0,1]")


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    check_probabilities("Metric probabilities", p)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "average_precision": float(average_precision_score(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, p)),
    }


def positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", np.array([0, 1])))
    positive = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2:
        raise RuntimeError("predict_proba did not return a two-dimensional array")
    if len(positive) == 1:
        return probabilities[:, int(positive[0])]
    if probabilities.shape[1] == 2:
        return probabilities[:, 1]
    raise RuntimeError("Could not identify the positive-class probability")


# =============================================================================
# 5. LOAD RAW DATA AND DEFINE THE EXACT OUTER FOLDS
# =============================================================================

for required_path in [
    TRAIN_FILE,
    TEST_FILE,
    IMPUTATION_MODELS_DIR,
    IMPUTATION_CLASSIFIER_SCENARIO_DIR,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required path does not exist: {required_path}")

print_section("LOADING RAW TRAIN / TEST")
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

if ID_COL not in train.columns or ID_COL not in test.columns or TARGET_COL not in train.columns:
    raise KeyError("Train/test must contain SK_ID_CURR; train must also contain TARGET")
if train[ID_COL].isna().any() or train[ID_COL].duplicated().any():
    raise ValueError("Train IDs must be unique and non-missing")
if test[ID_COL].isna().any() or test[ID_COL].duplicated().any():
    raise ValueError("Test IDs must be unique and non-missing")
if train[TARGET_COL].isna().any() or set(train[TARGET_COL].unique()) != {0, 1}:
    raise ValueError("TARGET must be non-missing and contain exactly {0,1}")
if set(train[ID_COL]).intersection(set(test[ID_COL])):
    raise ValueError("Train and test IDs overlap")

if SAMPLE_SUB_FILE.is_file():
    sample_sub = pd.read_csv(SAMPLE_SUB_FILE)
    if ID_COL not in sample_sub.columns or TARGET_COL not in sample_sub.columns:
        raise ValueError("sample_submission.csv must contain SK_ID_CURR and TARGET")
    if set(sample_sub[ID_COL]) != set(test[ID_COL]):
        raise ValueError("sample_submission IDs do not match test IDs")
else:
    sample_sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET_COL: np.nan})

y = train[TARGET_COL].astype(np.int32).to_numpy()
train_ids = train[ID_COL].to_numpy()
test_ids = test[ID_COL].to_numpy()
n_train, n_test = len(train), len(test)

feature_columns = [column for column in train.columns if column not in (ID_COL, TARGET_COL)]
test_feature_columns = [column for column in test.columns if column != ID_COL]
if set(feature_columns) != set(test_feature_columns):
    raise ValueError(
        "Train/test feature mismatch: "
        f"missing_in_test={sorted(set(feature_columns)-set(test_feature_columns))}; "
        f"extra_in_test={sorted(set(test_feature_columns)-set(feature_columns))}"
    )
test = test.loc[:, [ID_COL] + feature_columns].copy()

required_columns = MLP_RAW_CATEGORICAL_COLUMNS + [
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "AMT_INCOME_TOTAL",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "OWN_CAR_AGE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_ID_PUBLISH",
    *EXT_SOURCE_COLUMNS,
]
missing_required = sorted(set(required_columns) - set(feature_columns))
if missing_required:
    raise KeyError(f"Missing required predictors: {missing_required}")

reserved_raw_names = {
    "CREDIT_TERM",
    "GOODS_PRICE_TO_ANNUITY",
 
    FAMILLY_DHOUSE_FEATURE,
    LOAN_STRUCTURE_FEATURE,
    NEW_INTERACTION_FEATURE,
    *EXT_SOURCE_MISSING_COLUMNS,
}
collisions = sorted(reserved_raw_names.intersection(feature_columns))
if collisions:
    raise ValueError(f"Raw CSV already contains reserved generated columns: {collisions}")

raw_categorical_columns = [
    column
    for column in feature_columns
    if (
        pd.api.types.is_object_dtype(train[column].dtype)
        or pd.api.types.is_string_dtype(train[column].dtype)
        or isinstance(train[column].dtype, pd.CategoricalDtype)
    )
]
for column in MLP_RAW_CATEGORICAL_COLUMNS:
    if column not in raw_categorical_columns:
        raw_categorical_columns.append(column)

X_train_original = train.loc[:, feature_columns].copy()
X_test_original = test.loc[:, feature_columns].copy()
for column in feature_columns:
    if column in raw_categorical_columns:
        X_train_original[column] = X_train_original[column].astype("string")
        X_test_original[column] = X_test_original[column].astype("string")
    else:
        X_train_original[column] = pd.to_numeric(X_train_original[column], errors="coerce")
        X_test_original[column] = pd.to_numeric(X_test_original[column], errors="coerce")
        X_train_original[column] = X_train_original[column].replace([np.inf, -np.inf], np.nan)
        X_test_original[column] = X_test_original[column].replace([np.inf, -np.inf], np.nan)

for column in raw_categorical_columns:
    X_train_original[column] = X_train_original[column].astype("string")
    X_test_original[column] = X_test_original[column].astype("string")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
fold_splits = list(skf.split(X_train_original, y))

print(f"Train rows: {n_train:,} | Test rows: {n_test:,} | Raw predictors: {len(feature_columns)}")
print(f"Positive rows: {int(y.sum()):,} | Positive rate: {y.mean():.8%}")
print(f"Raw categorical columns: {len(raw_categorical_columns)}")


# =============================================================================
# 6. VERIFY AND LOAD THE PREVIOUSLY SAVED IMPUTATION ARTIFACTS
# =============================================================================

def load_fold_indices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        train_keys = [
            key for key in keys
            if ("train" in key.lower() or "fit" in key.lower())
            and "valid" not in key.lower()
        ]
        valid_keys = [key for key in keys if "valid" in key.lower()]
        if not train_keys or not valid_keys:
            raise KeyError(f"Could not identify train/valid arrays in {path}; keys={keys}")
        fit = np.asarray(archive[train_keys[0]], dtype=np.int64)
        valid = np.asarray(archive[valid_keys[0]], dtype=np.int64)
    if fit.ndim != 1 or valid.ndim != 1:
        raise ValueError(f"Fold indices must be one-dimensional: {path}")
    if len(np.unique(fit)) != len(fit) or len(np.unique(valid)) != len(valid):
        raise ValueError(f"Duplicate row indices found in {path}")
    if np.intersect1d(fit, valid).size:
        raise ValueError(f"Training/validation indices overlap in {path}")
    return fit, valid


print_section("VERIFYING SAVED IMPUTATION FOLDS")
IMPUTATION_INDEX_PATHS: dict[int, Path] = {}
IMPUTATION_VALUES_PATHS: dict[int, Path] = {}
IMPUTER_MODEL_PATHS: dict[tuple[int, str], Path] = {}

for fold_number, (fit_indices, valid_indices) in enumerate(fold_splits, start=1):
    indices_path = (
        IMPUTATION_CLASSIFIER_SCENARIO_DIR
        / "CHECKPOINTS"
        / "FOLDS"
        / f"FOLD_{fold_number:02d}"
        / "TRAIN_VALID_INDICES.npz"
    )
    values_path = (
        IMPUTATION_MODELS_DIR
        / f"FOLD_{fold_number:02d}"
        / "IMPUTED_VALUES.npz"
    )
    if not indices_path.is_file():
        raise FileNotFoundError(f"Saved imputation fold indices not found: {indices_path}")
    if not values_path.is_file():
        raise FileNotFoundError(f"Saved training imputed values not found: {values_path}")

    saved_fit, saved_valid = load_fold_indices(indices_path)
    if not np.array_equal(np.sort(fit_indices), np.sort(saved_fit)):
        raise RuntimeError(f"Fold {fold_number}: training indices differ from saved imputation experiment")
    if not np.array_equal(np.sort(valid_indices), np.sort(saved_valid)):
        raise RuntimeError(f"Fold {fold_number}: validation indices differ from saved imputation experiment")

    IMPUTATION_INDEX_PATHS[fold_number] = indices_path
    IMPUTATION_VALUES_PATHS[fold_number] = values_path

    for source in EXT_SOURCE_COLUMNS:
        model_path = (
            IMPUTATION_MODELS_DIR
            / f"FOLD_{fold_number:02d}"
            / source
            / "imputer.cbm"
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved {source} imputer not found: {model_path}")
        IMPUTER_MODEL_PATHS[(fold_number, source)] = model_path

    print(
        f"Fold {fold_number}: exact split verified; saved values and three imputers found.",
        flush=True,
    )


def load_and_validate_training_imputations(fold_number: int) -> dict[str, np.ndarray]:
    path = IMPUTATION_VALUES_PATHS[fold_number]
    with np.load(path, allow_pickle=False) as archive:
        missing_keys = sorted(set(EXT_SOURCE_COLUMNS) - set(archive.files))
        if missing_keys:
            raise KeyError(f"Fold {fold_number}: imputed-values file is missing {missing_keys}")
        result = {
            source: np.asarray(archive[source], dtype=np.float64)
            for source in EXT_SOURCE_COLUMNS
        }

    for source in EXT_SOURCE_COLUMNS:
        raw = pd.to_numeric(X_train_original[source], errors="coerce").to_numpy(dtype=np.float64)
        values = result[source]
        if values.shape != (n_train,):
            raise ValueError(f"Fold {fold_number} {source}: unexpected saved shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Fold {fold_number} {source}: saved imputation contains NaN/inf")
        observed = np.isfinite(raw)
        if not np.allclose(values[observed], raw[observed], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Fold {fold_number} {source}: saved imputation changed observed values")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"Fold {fold_number} {source}: saved values fall outside [0,1]")
    return result


def make_ratio_features_for_imputer(frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate temporary ratio features used by the saved imputer models."""
    output = frame.copy()
    annuity = pd.to_numeric(output["AMT_ANNUITY"], errors="coerce").replace(0, np.nan)
    credit = pd.to_numeric(output["AMT_CREDIT"], errors="coerce")
    goods = pd.to_numeric(output["AMT_GOODS_PRICE"], errors="coerce")
    output["CREDIT_TERM"] = (credit / annuity).replace([np.inf, -np.inf], np.nan)
    output["GOODS_PRICE_TO_ANNUITY"] = (goods / annuity).replace([np.inf, -np.inf], np.nan)
    return output


def build_imputer_input_frame(
    raw_frame: pd.DataFrame,
    target_source: str,
    feature_names: list[str],
    categorical_indices: list[int],
) -> pd.DataFrame:
    """Rebuild the saved CatBoost imputer's exact names and column ordering."""
    all_features = make_ratio_features_for_imputer(raw_frame)
    for source in EXT_SOURCE_COLUMNS:
        if source != target_source:
            all_features[f"{source}_MISSING"] = (
                pd.to_numeric(raw_frame[source], errors="coerce").isna().astype(np.int8)
            )

    missing_features = [name for name in feature_names if name not in all_features.columns]
    if missing_features:
        raise KeyError(
            f"Cannot rebuild {target_source} saved-imputer inputs; missing={missing_features}"
        )
    if target_source in feature_names:
        raise RuntimeError(f"{target_source} must not be included in its own imputer inputs")
    if "GOODS_PRICE_TO_ANNUITY" not in feature_names:
        raise RuntimeError(
            f"Saved {target_source} imputer does not include GOODS_PRICE_TO_ANNUITY; "
            "the selected imputation experiment may be incorrect."
        )
    if len(feature_names) != EXPECTED_SAVED_IMPUTER_FEATURE_COUNT:
        raise RuntimeError(
            f"Saved {target_source} imputer expected {EXPECTED_SAVED_IMPUTER_FEATURE_COUNT} "
            f"input features, found {len(feature_names)}"
        )

    categorical_indices = [int(index) for index in categorical_indices]
    if any(index < 0 or index >= len(feature_names) for index in categorical_indices):
        raise ValueError(f"Invalid categorical feature index in saved {target_source} imputer")
    categorical_names = {feature_names[index] for index in categorical_indices}

    result = all_features.loc[:, feature_names].copy()
    for column in feature_names:
        if column in categorical_names:
            result[column] = result[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce")
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
    return result


def generate_test_imputations(
    fold_number: int,
    fit_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Use each fold's saved imputer to fill missing TEST EXT_SOURCE values."""
    fold_dir = IMPUTATION_TEST_DIR / f"FOLD_{fold_number:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    values_path = fold_dir / "TEST_IMPUTED_VALUES.npz"
    audit_path = fold_dir / "TEST_IMPUTATION_AUDIT.csv"
    marker_path = fold_dir / "TEST_IMPUTATION_COMPLETE.json"
    required = [values_path, audit_path]

    if checkpoint_complete(marker_path, "test_imputation", CONFIG_HASH, required):
        with np.load(values_path, allow_pickle=False) as archive:
            restored = {
                source: np.asarray(archive[source], dtype=np.float64)
                for source in EXT_SOURCE_COLUMNS
            }
        for source in EXT_SOURCE_COLUMNS:
            raw = pd.to_numeric(X_test_original[source], errors="coerce").to_numpy(dtype=np.float64)
            values = restored[source]
            if values.shape != (n_test,) or not np.isfinite(values).all():
                raise RuntimeError(f"Fold {fold_number} {source}: restored test values are invalid")
            observed = np.isfinite(raw)
            if not np.allclose(values[observed], raw[observed], rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Fold {fold_number} {source}: restored test imputation changed observed values")
        print(f"Fold {fold_number}: restored TEST imputation checkpoint.", flush=True)
        return restored

    print_section(f"GENERATING TEST IMPUTATIONS — FOLD {fold_number}/{N_SPLITS}")
    output: dict[str, np.ndarray] = {}
    audit_rows = []

    for source in EXT_SOURCE_COLUMNS:
        model_path = IMPUTER_MODEL_PATHS[(fold_number, source)]
        imputer = CatBoostRegressor()
        imputer.load_model(str(model_path))
        feature_names = list(imputer.feature_names_)
        categorical_indices = list(imputer.get_cat_feature_indices())

        X_imp_test = build_imputer_input_frame(
            X_test_original,
            source,
            feature_names,
            categorical_indices,
        )
        raw = pd.to_numeric(X_test_original[source], errors="coerce").to_numpy(dtype=np.float64)
        missing = ~np.isfinite(raw)
        filled = raw.copy()

        fold_train_source = pd.to_numeric(
            X_train_original.iloc[fit_indices][source], errors="coerce"
        ).to_numpy(dtype=np.float64)
        fold_train_observed = fold_train_source[np.isfinite(fold_train_source)]
        if fold_train_observed.size == 0:
            raise RuntimeError(f"Fold {fold_number} {source}: no observed fold-training values")
        clip_min = float(fold_train_observed.min())
        clip_max = float(fold_train_observed.max())

        clipped_count = 0
        if missing.any():
            prediction = np.asarray(
                imputer.predict(X_imp_test.loc[missing]), dtype=np.float64
            ).reshape(-1)
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Fold {fold_number} {source}: saved imputer produced NaN/inf")
            clipped = np.clip(prediction, clip_min, clip_max)
            clipped_count = int(np.sum(clipped != prediction))
            filled[missing] = clipped

        if not np.isfinite(filled).all():
            raise RuntimeError(f"Fold {fold_number} {source}: TEST imputation is incomplete")
        output[source] = filled
        audit_rows.append(
            {
                "FOLD": fold_number,
                "SOURCE": source,
                "TEST_ROWS": n_test,
                "OBSERVED_TEST_VALUES": int((~missing).sum()),
                "MISSING_TEST_VALUES_IMPUTED": int(missing.sum()),
                "CLIPPED_PREDICTIONS": clipped_count,
                "CLIP_MIN_FROM_FOLD_TRAIN": clip_min,
                "CLIP_MAX_FROM_FOLD_TRAIN": clip_max,
                "IMPUTED_MIN": float(filled.min()),
                "IMPUTED_MAX": float(filled.max()),
                "MODEL_FEATURE_COUNT": len(feature_names),
                "GOODS_PRICE_TO_ANNUITY_IN_IMPUTER": True,
                "MODEL_CATEGORICAL_FEATURES": [feature_names[index] for index in categorical_indices],
            }
        )
        del imputer, X_imp_test
        gc.collect()

    atomic_npz(values_path, **output)
    atomic_csv(audit_path, pd.DataFrame(audit_rows))
    mark_complete(marker_path, "test_imputation", CONFIG_HASH, required)
    print(pd.DataFrame(audit_rows).to_string(index=False), flush=True)
    return output


# =============================================================================
# 7. EXT_SOURCE AUGMENTATION + TWO NEW REQUESTED FEATURES
# =============================================================================

def augment_with_ext_source(
    original_frame: pd.DataFrame,
    imputed_values: dict[str, np.ndarray],
) -> pd.DataFrame:
    output = original_frame.copy()
    for source in EXT_SOURCE_COLUMNS:
        raw = pd.to_numeric(original_frame[source], errors="coerce").to_numpy(dtype=np.float64)
        imputed = np.asarray(imputed_values[source], dtype=np.float64)
        if imputed.shape != (len(original_frame),) or not np.isfinite(imputed).all():
            raise ValueError(f"Invalid imputed vector for {source}")
        missing = ~np.isfinite(raw)
        final_values = raw.copy()
        final_values[missing] = imputed[missing]
        if not np.isfinite(final_values).all():
            raise RuntimeError(f"{source}: missing values remain after saved imputation")
        output[source] = final_values.astype(np.float32)
        output[f"{source}_MISSING"] = missing.astype(np.int8)
    return output


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)



def familly_dhouse_keys(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"FAMILLY_DHOUSE source columns are missing: {missing}")
    return pd.DataFrame(
        {
            "GENDER": frame["CODE_GENDER"].astype("string").fillna(MISSING_CATEGORY).astype(str),
            "FAMILY_STATUS": frame["NAME_FAMILY_STATUS"].astype("string").fillna(MISSING_CATEGORY).astype(str),
            "HOUSING_TYPE": frame["NAME_HOUSING_TYPE"].astype("string").fillna(MISSING_CATEGORY).astype(str),
        },
        index=frame.index,
    )


def combination_counts(
    reference: pd.DataFrame,
    query: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    count_map = reference.groupby(columns, dropna=False, sort=False).size().to_dict()
    rows = query[columns].itertuples(index=False, name=None)
    return np.fromiter(
        (int(count_map.get(tuple(row), 0)) for row in rows),
        dtype=np.int64,
        count=len(query),
    )


def build_familly_dhouse_profile(
    reference_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    min_count: int = FAMILLY_DHOUSE_MIN_COUNT,
) -> pd.Series:
    """
    Hierarchical FAMILLY_DHOUSE using counts from reference_frame only.

    Priority:
      1. GENDER + FAMILY_STATUS + HOUSING_TYPE if count >= 40
      2. GENDER + FAMILY_STATUS if count >= 40
      3. GENDER + HOUSING_TYPE if count >= 40
      4. GENDER only (fallback)

    This function never reads TARGET. For fold-training rows the reference is
    the fold-training partition itself; validation and TEST are looked up
    against that same training-only reference.
    """
    if int(min_count) < 1:
        raise ValueError("FAMILLY_DHOUSE min_count must be >= 1")

    reference_keys = familly_dhouse_keys(reference_frame)
    query_keys = familly_dhouse_keys(query_frame)

    triple_count = combination_counts(
        reference_keys, query_keys, ["GENDER", "FAMILY_STATUS", "HOUSING_TYPE"]
    )
    gender_family_count = combination_counts(
        reference_keys, query_keys, ["GENDER", "FAMILY_STATUS"]
    )
    gender_housing_count = combination_counts(
        reference_keys, query_keys, ["GENDER", "HOUSING_TYPE"]
    )

    gender = query_keys["GENDER"].to_numpy(dtype=object)
    family = query_keys["FAMILY_STATUS"].to_numpy(dtype=object)
    housing = query_keys["HOUSING_TYPE"].to_numpy(dtype=object)

    values = np.asarray([f"GENDER::{value}" for value in gender], dtype=object)
    use_triple = triple_count >= int(min_count)
    use_gender_family = (~use_triple) & (gender_family_count >= int(min_count))
    use_gender_housing = (
        (~use_triple)
        & (~use_gender_family)
        & (gender_housing_count >= int(min_count))
    )

    for index in np.flatnonzero(use_triple):
        values[index] = (
            f"GENDER_FAMILY_STATUS_HOUSING_TYPE::"
            f"{gender[index]}::{family[index]}::{housing[index]}"
        )
    for index in np.flatnonzero(use_gender_family):
        values[index] = f"GENDER_FAMILY_STATUS::{gender[index]}::{family[index]}"
    for index in np.flatnonzero(use_gender_housing):
        values[index] = f"GENDER_HOUSING_TYPE::{gender[index]}::{housing[index]}"

    return pd.Series(values, index=query_frame.index, name=FAMILLY_DHOUSE_FEATURE, dtype="string")


# =============================================================================
# 8. FOLD-LOCAL NEIGHBOUR INTERACTION
# =============================================================================

def extract_ext_source_sum_and_age(
    frame: pd.DataFrame,
    age_median: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    source_matrix = np.column_stack(
        [pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
         for column in EXT_SOURCE_COLUMNS]
    )
    if not np.isfinite(source_matrix).all():
        raise ValueError("EXT_SOURCE values must be imputed before neighbour interaction")
    ext_sum = source_matrix.sum(axis=1)
    days_birth = pd.to_numeric(frame["DAYS_BIRTH"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    age = (-days_birth / 365.25).to_numpy(dtype=np.float64)
    if age_median is None:
        finite = age[np.isfinite(age)]
        if finite.size == 0:
            raise ValueError("No finite DAYS_BIRTH values in fold-training rows")
        age_median = float(np.median(finite))
    age = np.where(np.isfinite(age), age, float(age_median))
    return ext_sum, age, float(age_median)


def compute_training_neighbor_means(
    coords: np.ndarray,
    reference_ext: np.ndarray,
    reference_age: np.ndarray,
    neighbor_model: NearestNeighbors,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(reference_ext)
    if n <= N_NEIGHBORS:
        raise ValueError(f"Need >{N_NEIGHBORS} fold-training rows; received {n}")
    mean_ext = np.empty(n, dtype=np.float64)
    mean_age = np.empty(n, dtype=np.float64)
    n_query = min(N_NEIGHBORS + 2, n)
    for start in range(0, n, NEIGHBOR_QUERY_CHUNK_SIZE):
        stop = min(start + NEIGHBOR_QUERY_CHUNK_SIZE, n)
        neighbors = neighbor_model.kneighbors(
            coords[start:stop], n_neighbors=n_query, return_distance=False
        )
        self_indices = np.arange(start, stop, dtype=np.int64)
        self_match = neighbors == self_indices[:, None]
        has_self = self_match.any(axis=1)
        self_position = np.where(has_self, self_match.argmax(axis=1), n_query)
        output_columns = np.arange(N_NEIGHBORS, dtype=np.int64)[None, :]
        source_columns = output_columns + (output_columns >= self_position[:, None])
        selected = np.take_along_axis(neighbors, source_columns, axis=1)
        if selected.shape[1] != N_NEIGHBORS:
            raise RuntimeError("Training-neighbour selection did not return exactly 500 neighbours")
        mean_ext[start:stop] = reference_ext[selected].mean(axis=1)
        mean_age[start:stop] = reference_age[selected].mean(axis=1)
    return mean_ext, mean_age


def compute_query_neighbor_means(
    coords: np.ndarray,
    reference_ext: np.ndarray,
    reference_age: np.ndarray,
    neighbor_model: NearestNeighbors,
) -> tuple[np.ndarray, np.ndarray]:
    mean_ext = np.empty(len(coords), dtype=np.float64)
    mean_age = np.empty(len(coords), dtype=np.float64)
    for start in range(0, len(coords), NEIGHBOR_QUERY_CHUNK_SIZE):
        stop = min(start + NEIGHBOR_QUERY_CHUNK_SIZE, len(coords))
        neighbors = neighbor_model.kneighbors(
            coords[start:stop], n_neighbors=N_NEIGHBORS, return_distance=False
        )
        mean_ext[start:stop] = reference_ext[neighbors].mean(axis=1)
        mean_age[start:stop] = reference_age[neighbors].mean(axis=1)
    return mean_ext, mean_age


def build_fold_local_neighbor_interaction(
    X_fit_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, pd.DataFrame]]:
    fit_ext, fit_age, age_median = extract_ext_source_sum_and_age(X_fit_raw)
    valid_ext, valid_age, _ = extract_ext_source_sum_and_age(X_valid_raw, age_median)
    test_ext, test_age, _ = extract_ext_source_sum_and_age(X_test_raw, age_median)

    fit_2d = np.column_stack([fit_ext, fit_age]).astype(np.float64)
    center = fit_2d.mean(axis=0)
    scale = fit_2d.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    fit_coords = (fit_2d - center) / scale
    valid_coords = (np.column_stack([valid_ext, valid_age]) - center) / scale
    test_coords = (np.column_stack([test_ext, test_age]) - center) / scale

    neighbor_model = NearestNeighbors(
        algorithm="kd_tree", metric="euclidean", leaf_size=40, n_jobs=CPU_THREADS
    )
    neighbor_model.fit(fit_coords)
    fit_mean_ext, fit_mean_age = compute_training_neighbor_means(
        fit_coords, fit_ext, fit_age, neighbor_model
    )
    valid_mean_ext, valid_mean_age = compute_query_neighbor_means(
        valid_coords, fit_ext, fit_age, neighbor_model
    )
    test_mean_ext, test_mean_age = compute_query_neighbor_means(
        test_coords, fit_ext, fit_age, neighbor_model
    )

    fit_interaction = (fit_ext - fit_mean_ext) * (fit_age - fit_mean_age)
    valid_interaction = (valid_ext - valid_mean_ext) * (valid_age - valid_mean_age)
    test_interaction = (test_ext - test_mean_ext) * (test_age - test_mean_age)
    for name, values in [
        ("fit", fit_interaction),
        ("validation", valid_interaction),
        ("test", test_interaction),
    ]:
        if not np.isfinite(values).all():
            raise RuntimeError(f"Local interaction contains NaN/inf in {name}")

    state = {
        "age_median": age_median,
        "center": center,
        "scale": scale,
        "neighbor_model": neighbor_model,
        "reference_ext_sum": fit_ext,
        "reference_age": fit_age,
        "n_neighbors": N_NEIGHBORS,
    }
    diagnostics = {
        "validation": pd.DataFrame(
            {
                "EXT_SOURCE_SUM_IMPUTED": valid_ext,
                "AGE_YEARS": valid_age,
                "MEAN_EXT_SOURCE_SUM_500_NEIGHBORS": valid_mean_ext,
                "MEAN_AGE_YEARS_500_NEIGHBORS": valid_mean_age,
                NEW_INTERACTION_FEATURE: valid_interaction,
            }
        ),
        "test": pd.DataFrame(
            {
                "EXT_SOURCE_SUM_IMPUTED": test_ext,
                "AGE_YEARS": test_age,
                "MEAN_EXT_SOURCE_SUM_500_NEIGHBORS": test_mean_ext,
                "MEAN_AGE_YEARS_500_NEIGHBORS": test_mean_age,
                NEW_INTERACTION_FEATURE: test_interaction,
            }
        ),
    }
    return fit_interaction, valid_interaction, test_interaction, state, diagnostics


# =============================================================================
# 9. FOLD-LOCAL HYBRID GMM SEGMENTER
# =============================================================================

class HybridLoanStructureSegmenter:
    """Fold-local target-free spike handling + BIC-selected Gaussian mixtures."""

    def __init__(self, fold_number: int):
        self.fold_number = int(fold_number)
        self.fitted = False

    @staticmethod
    def calculate_ratios(frame: pd.DataFrame):
        credit = pd.to_numeric(frame["AMT_CREDIT"], errors="coerce")
        annuity = pd.to_numeric(frame["AMT_ANNUITY"], errors="coerce")
        goods_price = pd.to_numeric(frame["AMT_GOODS_PRICE"], errors="coerce")
        annuity = annuity.replace([np.inf, -np.inf], np.nan).where(annuity > 0)
        goods_price = goods_price.replace([np.inf, -np.inf], np.nan).where(goods_price > 0)
        term = (credit / annuity).replace([np.inf, -np.inf], np.nan).where(lambda values: values > 0)
        goods = (credit / goods_price).replace([np.inf, -np.inf], np.nan).where(lambda values: values > 0)
        return term, goods

    @staticmethod
    def detect_spike(values: np.ndarray, precision: int, axis_name: str) -> dict[str, Any]:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            return {
                "value": None, "share": 0.0, "rounded_top_count": 0,
                "second_count": 0, "valid_count": 0, "detected": False,
                "axis": axis_name, "precision": int(precision),
            }
        counts = pd.Series(np.round(values, precision)).value_counts()
        top_value = float(counts.index[0])
        top_count = int(counts.iloc[0])
        second_count = int(counts.iloc[1]) if len(counts) > 1 else 0
        share = top_count / len(values)
        detected = bool(
            len(counts) >= 2
            and share >= LOAN_STRUCTURE_SPIKE_MIN_SHARE
            and top_count >= LOAN_STRUCTURE_SPIKE_DOMINANCE * second_count
        )
        return {
            "value": top_value if detected else None,
            "share": float(share if detected else 0.0),
            "rounded_top_value": top_value,
            "rounded_top_count": top_count,
            "second_count": second_count,
            "valid_count": int(len(values)),
            "detected": detected,
            "axis": axis_name,
            "precision": int(precision),
        }

    @staticmethod
    def spike_mask(values: np.ndarray, info: dict[str, Any], precision: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values) & (values > 0)
        if info["value"] is None:
            return np.zeros(len(values), dtype=bool)
        return valid & (np.round(values, precision) == info["value"])

    def fit_gmm_by_bic(self, X_raw: np.ndarray, branch_name: str):
        X_raw = np.asarray(X_raw, dtype=np.float64)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(-1, 1)
        if X_raw.ndim != 2 or X_raw.shape[1] not in (1, 2) or not np.isfinite(X_raw).all():
            raise ValueError(f"{branch_name}: invalid GMM input")
        if len(X_raw) < GMM_MIN_COMPONENT_ROWS:
            raise ValueError(f"{branch_name}: insufficient rows")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        max_k = min(GMM_MAX_COMPONENTS, max(1, len(X_scaled) // GMM_MIN_COMPONENT_ROWS))
        search_rows, size_rows = [], []
        best_model, best_bic, best_counts, best_k = None, np.inf, None, None

        for k in range(1, max_k + 1):
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=GMM_REG_COVAR,
                max_iter=GMM_MAX_ITER,
                n_init=GMM_N_INIT,
                init_params="kmeans",
                random_state=GMM_RANDOM_STATE + 100 * self.fold_number + k,
            )
            model.fit(X_scaled)
            assignments = model.predict(X_scaled)
            counts = np.bincount(assignments, minlength=k)
            bic = float(model.bic(X_scaled))
            eligible = bool(model.converged_ and int(counts.min()) >= GMM_MIN_COMPONENT_ROWS)
            centers = scaler.inverse_transform(model.means_)
            order = np.argsort(centers[:, 0]) if centers.shape[1] == 1 else np.lexsort((centers[:, 1], centers[:, 0]))
            search_rows.append(
                {
                    "branch": branch_name,
                    "n_components": int(k),
                    "bic": bic,
                    "converged": bool(model.converged_),
                    "minimum_hard_cluster_rows": int(counts.min()),
                    "maximum_hard_cluster_rows": int(counts.max()),
                    "eligible_for_selection": eligible,
                    "is_search_boundary": bool(k == max_k),
                    "iterations": int(model.n_iter_),
                }
            )
            for rank, old in enumerate(order, start=1):
                center = centers[old]
                size_rows.append(
                    {
                        "branch": branch_name,
                        "n_components": int(k),
                        "component_rank_by_center": int(rank),
                        "original_model_component": int(old),
                        "hard_count": int(counts[old]),
                        "mixture_weight": float(model.weights_[old]),
                        "mean_log_feature_1": float(center[0]),
                        "mean_log_feature_2": float(center[1]) if len(center) > 1 else np.nan,
                        "selected_by_bic": False,
                    }
                )
            if eligible and bic < best_bic:
                best_model, best_bic, best_counts, best_k = model, bic, counts.copy(), k
            else:
                del model
            del assignments, counts

        if best_model is None:
            raise RuntimeError(f"{branch_name}: no eligible converged GMM found")
        for row in size_rows:
            row["selected_by_bic"] = int(row["n_components"]) == int(best_k)

        centers = scaler.inverse_transform(best_model.means_)
        order = np.argsort(centers[:, 0]) if centers.shape[1] == 1 else np.lexsort((centers[:, 1], centers[:, 0]))
        rank_map = np.empty(best_model.n_components, dtype=np.int32)
        components = []
        for rank, old in enumerate(order, start=1):
            rank_map[int(old)] = rank
            center = centers[old]
            components.append(
                {
                    "ranked_component": int(rank),
                    "original_model_component": int(old),
                    "weight": float(best_model.weights_[old]),
                    "training_hard_count": int(best_counts[old]),
                    "geometric_center_credit_term": float(np.exp(center[0])),
                    "geometric_center_credit_to_goods": float(np.exp(center[1])) if len(center) > 1 else None,
                }
            )

        reference_assignments = best_model.predict(X_scaled)
        stability_rows, aris = [], []
        for repeat in range(1, GMM_STABILITY_REPEATS + 1):
            restart = GaussianMixture(
                n_components=best_k,
                covariance_type="full",
                reg_covar=GMM_REG_COVAR,
                max_iter=GMM_MAX_ITER,
                n_init=1,
                init_params="kmeans",
                random_state=GMM_RANDOM_STATE + 900000 + 1000 * self.fold_number + 100 * best_k + repeat,
            )
            restart.fit(X_scaled)
            restart_labels = restart.predict(X_scaled)
            restart_counts = np.bincount(restart_labels, minlength=best_k)
            ari = float(adjusted_rand_score(reference_assignments, restart_labels))
            aris.append(ari)
            stability_rows.append(
                {
                    "repeat": int(repeat),
                    "converged": bool(restart.converged_),
                    "iterations": int(restart.n_iter_),
                    "adjusted_rand_vs_selected": ari,
                    "minimum_hard_cluster_rows": int(restart_counts.min()),
                    "maximum_hard_cluster_rows": int(restart_counts.max()),
                }
            )
            del restart, restart_labels, restart_counts

        metadata = {
            "branch": branch_name,
            "input_dimensions": int(X_raw.shape[1]),
            "training_rows": int(len(X_raw)),
            "selected_n_components": int(best_k),
            "selected_bic": float(best_bic),
            "selected_at_search_boundary": bool(best_k == max_k),
            "components_sorted_by_center": components,
            "bic_search": search_rows,
            "candidate_component_sizes": size_rows,
            "initialization_stability": {
                "repeats": GMM_STABILITY_REPEATS,
                "mean_ari_vs_selected": float(np.mean(aris)),
                "minimum_ari_vs_selected": float(np.min(aris)),
                "maximum_ari_vs_selected": float(np.max(aris)),
                "restart_details": stability_rows,
            },
        }
        del reference_assignments, X_scaled
        return scaler, best_model, rank_map, metadata

    def fit(self, frame: pd.DataFrame):
        term, goods = self.calculate_ratios(frame)
        t = term.to_numpy(dtype=np.float64)
        g = goods.to_numpy(dtype=np.float64)
        term_valid = np.isfinite(t) & (t > 0)
        goods_valid = np.isfinite(g) & (g > 0)
        self.term_spike_info = self.detect_spike(t, LOAN_STRUCTURE_TERM_PRECISION, "CREDIT_TERM")
        self.goods_spike_info = self.detect_spike(g, LOAN_STRUCTURE_GOODS_PRECISION, "CREDIT_TO_GOODS")
        term_spike = self.spike_mask(t, self.term_spike_info, LOAN_STRUCTURE_TERM_PRECISION)
        goods_spike = self.spike_mask(g, self.goods_spike_info, LOAN_STRUCTURE_GOODS_PRECISION)

        both_valid = term_valid & goods_valid
        both = both_valid & term_spike & goods_spike
        term_only = both_valid & term_spike & ~goods_spike
        goods_only = both_valid & ~term_spike & goods_spike
        neither = both_valid & ~term_spike & ~goods_spike
        self.training_branch_counts = {
            "BOTH_SPIKES": int(both.sum()),
            "TERM_SPIKE_ONLY": int(term_only.sum()),
            "GOODS_SPIKE_ONLY": int(goods_only.sum()),
            "NEITHER_SPIKE_2D": int(neither.sum()),
            "MISSING_TERM": int((~term_valid & goods_valid).sum()),
            "MISSING_GOODS": int((term_valid & ~goods_valid).sum()),
            "MISSING_BOTH": int((~term_valid & ~goods_valid).sum()),
        }
        if goods_only.sum() < GMM_MIN_COMPONENT_ROWS or neither.sum() < GMM_MIN_COMPONENT_ROWS:
            raise RuntimeError(
                f"Fold {self.fold_number}: insufficient rows in GMM branches: "
                f"{self.training_branch_counts}"
            )

        (
            self.term_gmm_scaler,
            self.term_gmm_model,
            self.term_gmm_rank_map,
            self.term_gmm_metadata,
        ) = self.fit_gmm_by_bic(
            np.log(t[goods_only]).reshape(-1, 1),
            "GOODS_SPIKE_ONLY__1D_GMM_ON_LOG_CREDIT_TERM",
        )
        (
            self.joint_gmm_scaler,
            self.joint_gmm_model,
            self.joint_gmm_rank_map,
            self.joint_gmm_metadata,
        ) = self.fit_gmm_by_bic(
            np.column_stack([np.log(t[neither]), np.log(g[neither])]),
            "NEITHER_SPIKE__2D_GMM_ON_BOTH_LOG_RATIOS",
        )
        self.training_support = {
            **self.training_branch_counts,
            "TERM_SPIKE_ONLY_REVIEW_THRESHOLD": TERM_SPIKE_ONLY_REVIEW_SUPPORT,
            "TERM_SPIKE_ONLY_HAS_REVIEW_SUPPORT": bool(term_only.sum() >= TERM_SPIKE_ONLY_REVIEW_SUPPORT),
        }
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("GMM segmenter must be fitted before transform")
        term, goods = self.calculate_ratios(frame)
        t = term.to_numpy(dtype=np.float64)
        g = goods.to_numpy(dtype=np.float64)
        term_valid = np.isfinite(t) & (t > 0)
        goods_valid = np.isfinite(g) & (g > 0)
        term_spike = self.spike_mask(t, self.term_spike_info, LOAN_STRUCTURE_TERM_PRECISION)
        goods_spike = self.spike_mask(g, self.goods_spike_info, LOAN_STRUCTURE_GOODS_PRECISION)

        valid = term_valid & goods_valid
        both = valid & term_spike & goods_spike
        term_only = valid & term_spike & ~goods_spike
        goods_only = valid & ~term_spike & goods_spike
        neither = valid & ~term_spike & ~goods_spike

        labels = np.full(len(frame), "MISSING_BOTH", dtype=object)
        labels[~term_valid & goods_valid] = "MISSING_TERM"
        labels[term_valid & ~goods_valid] = "MISSING_GOODS"
        labels[both] = "BOTH_SPIKES"
        labels[term_only] = "TERM_SPIKE_ONLY"
        if goods_only.any():
            scaled = self.term_gmm_scaler.transform(np.log(t[goods_only]).reshape(-1, 1))
            ranks = self.term_gmm_rank_map[self.term_gmm_model.predict(scaled)]
            labels[goods_only] = [f"GOODS_SPIKE_TERM_GMM_{rank:02d}" for rank in ranks]
        if neither.any():
            scaled = self.joint_gmm_scaler.transform(
                np.column_stack([np.log(t[neither]), np.log(g[neither])])
            )
            ranks = self.joint_gmm_rank_map[self.joint_gmm_model.predict(scaled)]
            labels[neither] = [f"GMM2D_{rank:02d}" for rank in ranks]
        return pd.Series(labels, index=frame.index, name=LOAN_STRUCTURE_FEATURE, dtype="object")

    def metadata(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot export metadata for an unfitted GMM segmenter")
        return {
            "fold": self.fold_number,
            "feature_name": LOAN_STRUCTURE_FEATURE,
            "segmentation_method": "Hybrid spike handling + BIC-selected GaussianMixture",
            "target_used": False,
            "training_branch_counts": self.training_branch_counts,
            "training_support": self.training_support,
            "term_spike_info": self.term_spike_info,
            "goods_spike_info": self.goods_spike_info,
            "goods_spike_only_1d_gmm": self.term_gmm_metadata,
            "neither_spike_2d_gmm": self.joint_gmm_metadata,
        }


# =============================================================================
# 9.1. PERSISTENT GMM CACHE — FOLD / INPUT / SETTINGS FINGERPRINT
# =============================================================================

def build_gmm_cache_key(
    fold_number: int,
    fit_indices: np.ndarray,
    X_train_raw: pd.DataFrame,
    train_id_values: np.ndarray,
) -> str:
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    train_id_values = np.asarray(train_id_values)
    missing = [column for column in GMM_CACHE_INPUT_COLUMNS if column not in X_train_raw.columns]
    if missing:
        raise KeyError(f"GMM fingerprint is missing columns: {missing}")

    source = X_train_raw.iloc[fit_indices][GMM_CACHE_INPUT_COLUMNS].copy()
    identity = pd.DataFrame({ID_COL: train_id_values[fit_indices]})
    for column in GMM_CACHE_INPUT_COLUMNS:
        identity[column] = source[column].to_numpy()

    configuration = {
        "cache_schema_version": GMM_CACHE_SCHEMA_VERSION,
        "fold_number": int(fold_number),
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "input_columns": GMM_CACHE_INPUT_COLUMNS,
        "spike_detection": {
            "term_precision": LOAN_STRUCTURE_TERM_PRECISION,
            "goods_precision": LOAN_STRUCTURE_GOODS_PRECISION,
            "minimum_share": LOAN_STRUCTURE_SPIKE_MIN_SHARE,
            "spike_dominance": LOAN_STRUCTURE_SPIKE_DOMINANCE,
            "term_spike_only_review_support": TERM_SPIKE_ONLY_REVIEW_SUPPORT,
        },
        "gmm": {
            "max_components": GMM_MAX_COMPONENTS,
            "n_init": GMM_N_INIT,
            "max_iter": GMM_MAX_ITER,
            "reg_covar": GMM_REG_COVAR,
            "minimum_component_rows": GMM_MIN_COMPONENT_ROWS,
            "random_state": GMM_RANDOM_STATE,
            "stability_repeats": GMM_STABILITY_REPEATS,
            "covariance_type": "full",
            "init_params": "kmeans",
        },
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    digest = hashlib.sha256()
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest.update(encoded)
    digest.update(fit_indices.astype("<i8", copy=False).tobytes())
    row_hashes = pd.util.hash_pandas_object(
        identity, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def load_matching_persistent_gmm(
    cache_key: str,
    fold_number: int,
) -> tuple[HybridLoanStructureSegmenter | None, Path | None]:
    """Scan cache files so the previous script's filename convention is not assumed."""
    for cache_path in sorted(GMM_CACHE_DIR.rglob("*.pkl")):
        try:
            payload = load_pickle(cache_path)
        except Exception as exc:
            print(f"[GMM CACHE WARNING] Cannot read {cache_path.name}: {exc}", flush=True)
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("cache_schema_version") != GMM_CACHE_SCHEMA_VERSION:
            continue
        if payload.get("cache_key") != cache_key:
            continue
        if int(payload.get("fold_number", -1)) != int(fold_number):
            continue
        segmenter = payload.get("segmenter")
        if (
            isinstance(segmenter, HybridLoanStructureSegmenter)
            and getattr(segmenter, "fitted", False)
            and int(segmenter.fold_number) == int(fold_number)
        ):
            print(f"[GMM CACHE HIT] Fold {fold_number}: {cache_path}", flush=True)
            return segmenter, cache_path
    return None, None


def save_persistent_gmm(
    segmenter: HybridLoanStructureSegmenter,
    cache_key: str,
) -> Path:
    if not segmenter.fitted:
        raise RuntimeError("Cannot cache an unfitted GMM segmenter")
    path = GMM_CACHE_DIR / f"FOLD_{segmenter.fold_number:02d}_{cache_key}.pkl"
    atomic_pickle(
        path,
        {
            "cache_schema_version": GMM_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "fold_number": int(segmenter.fold_number),
            "segmenter": segmenter,
        },
    )
    print(f"[GMM CACHE SAVED] Fold {segmenter.fold_number}: {path}", flush=True)
    return path


def get_fold_gmm_segmenter(
    fold_number: int,
    fit_indices: np.ndarray,
    X_train_raw: pd.DataFrame,
    train_id_values: np.ndarray,
) -> tuple[HybridLoanStructureSegmenter, dict[str, Any], str]:
    """Restore run checkpoint, else persistent cache, else fit and cache once."""
    run_fold_dir = GMM_DIR / f"FOLD_{fold_number:02d}"
    run_fold_dir.mkdir(parents=True, exist_ok=True)
    run_model_path = run_fold_dir / "segmenter.pkl"
    metadata_path = run_fold_dir / "segmenter_metadata.json"
    status_path = run_fold_dir / "GMM_CACHE_STATUS.json"
    run_marker = run_fold_dir / "GMM_COMPLETE.json"
    cache_key = build_gmm_cache_key(fold_number, fit_indices, X_train_raw, train_id_values)

    if checkpoint_complete(
        run_marker,
        "gmm_segmenter",
        CONFIG_HASH,
        [run_model_path, metadata_path, status_path],
    ):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("cache_key") == cache_key:
            segmenter = load_pickle(run_model_path)
            if (
                isinstance(segmenter, HybridLoanStructureSegmenter)
                and getattr(segmenter, "fitted", False)
                and segmenter.fold_number == fold_number
            ):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                print(f"Fold {fold_number}: restored run-specific GMM checkpoint.", flush=True)
                return segmenter, metadata, "RUN_CHECKPOINT"
        print(f"Fold {fold_number}: run GMM checkpoint has a different fingerprint; checking persistent cache.", flush=True)

    segmenter, persistent_path = load_matching_persistent_gmm(cache_key, fold_number)
    if segmenter is not None:
        cache_source = "PERSISTENT_CACHE"
    else:
        print_section(f"FITTING NEW FOLD-LOCAL GMM — FOLD {fold_number}")
        segmenter = HybridLoanStructureSegmenter(fold_number)
        segmenter.fit(X_train_raw.iloc[fit_indices])
        persistent_path = save_persistent_gmm(segmenter, cache_key)
        cache_source = "FITTED_AND_CACHED"

    metadata = segmenter.metadata()
    atomic_pickle(run_model_path, segmenter)
    atomic_json(metadata_path, metadata)
    atomic_json(
        status_path,
        {
            "fold": int(fold_number),
            "cache_key": cache_key,
            "cache_source": cache_source,
            "persistent_cache_path": str(persistent_path) if persistent_path else None,
            "run_model_path": str(run_model_path),
            "training_rows": int(len(fit_indices)),
            "validation_rows": int(n_train - len(fit_indices)),
            "selected_1d_k": int(segmenter.term_gmm_metadata["selected_n_components"]),
            "selected_2d_k": int(segmenter.joint_gmm_metadata["selected_n_components"]),
        },
    )
    mark_complete(
        run_marker,
        "gmm_segmenter",
        CONFIG_HASH,
        [run_model_path, metadata_path, status_path],
    )
    return segmenter, metadata, cache_source


# =============================================================================
# 10. TREE / REALMLP FEATURE ENGINEERING
# =============================================================================

def fit_tree_feature_statistics(X_fit: pd.DataFrame) -> dict[str, pd.Series]:
    income = pd.to_numeric(X_fit["AMT_INCOME_TOTAL"], errors="coerce")
    region = pd.to_numeric(X_fit["REGION_RATING_CLIENT"], errors="coerce")
    means = (
        pd.DataFrame({"REGION": region, "INCOME": income})
        .dropna(subset=["REGION"])
        .groupby("REGION")["INCOME"]
        .mean()
    )
    return {"regional_means": means}


def add_tree_engineered_features(
    frame: pd.DataFrame,
    statistics: dict[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
    credit = pd.to_numeric(output["AMT_CREDIT"], errors="coerce")
    annuity = pd.to_numeric(output["AMT_ANNUITY"], errors="coerce").replace(0, np.nan)
    goods = pd.to_numeric(output["AMT_GOODS_PRICE"], errors="coerce").replace(0, np.nan)
    income = pd.to_numeric(output["AMT_INCOME_TOTAL"], errors="coerce")
    region = pd.to_numeric(output["REGION_RATING_CLIENT"], errors="coerce")
    output["CREDIT_TERM"] = safe_ratio(credit, annuity)
    region_mean = region.map(statistics["regional_means"]).replace(0, np.nan)
    output["INCOME_TO_REGION_MEAN"] = safe_ratio(income, region_mean)
    output["OWN_CAR_AGE_CLEAN"] = (
        pd.to_numeric(output["OWN_CAR_AGE"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-1.0)
    )
    # Preserve the original four-model pipeline's squared DIFF_RATIO definition.
    output["DIFF_RATIO"] = ((credit - goods) / goods) ** 2
    numeric_added = [
        "CREDIT_TERM",
        "INCOME_TO_REGION_MEAN",
        "OWN_CAR_AGE_CLEAN",
        "DIFF_RATIO",
       
    ]
    output[numeric_added] = output[numeric_added].replace([np.inf, -np.inf], np.nan)
    return output


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(np.float64)


def fit_mlp_region_income_means(reference_df: pd.DataFrame) -> tuple[pd.Series, float]:
    income = numeric_column(reference_df, "AMT_INCOME_TOTAL")
    region = reference_df["REGION_RATING_CLIENT"].astype("string").fillna("__MISSING_REGION__")
    means = pd.DataFrame({"region": region.to_numpy(), "income": income.to_numpy()}).groupby(
        "region", dropna=False
    )["income"].mean()
    global_mean = float(income.mean())
    if not np.isfinite(global_mean) or global_mean == 0:
        global_mean = 1.0
    return means, global_mean


def build_mlp_feature_frame(
    query_df: pd.DataFrame,
    region_means: pd.Series,
    global_income_mean: float,
    local_interaction: np.ndarray,
    loan_cluster: pd.Series,
    raw_feature_columns: list[str],
) -> pd.DataFrame:
    output = query_df[raw_feature_columns].copy()
    credit = numeric_column(query_df, "AMT_CREDIT")
    annuity = numeric_column(query_df, "AMT_ANNUITY")
    income = numeric_column(query_df, "AMT_INCOME_TOTAL")
    goods = numeric_column(query_df, "AMT_GOODS_PRICE")

    output["CREDIT_TERM"] = safe_ratio(credit, annuity)
    region_key = query_df["REGION_RATING_CLIENT"].astype("string").fillna("__MISSING_REGION__")
    region_mean = region_key.map(region_means).astype(np.float64).fillna(global_income_mean)
    output["INCOME_TO_REGION_MEAN"] = safe_ratio(income, region_mean)
    output["OWN_CAR_AGE_CLEAN"] = numeric_column(query_df, "OWN_CAR_AGE").fillna(-1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        output["DIFF_RATIO"] = ((credit - goods) / goods) ** 2
    output["DIFF_RATIO"] = output["DIFF_RATIO"].replace([np.inf, -np.inf], np.nan)
  
    if len(local_interaction) != len(output) or len(loan_cluster) != len(output):
        raise ValueError("RealMLP engineered arrays have incorrect row counts")
    output[NEW_INTERACTION_FEATURE] = np.asarray(local_interaction, dtype=np.float32)
    output[LOAN_STRUCTURE_FEATURE] = loan_cluster.astype("string")
    numeric_columns = [column for column in output.columns if column not in MLP_CATEGORICAL_COLUMNS]
    output[numeric_columns] = output[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return output


def encode_mlp_fold_inputs(
    X_fit_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    X_test_raw_fold: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for column in MLP_CATEGORICAL_COLUMNS:
        if column not in X_fit_raw.columns:
            raise KeyError(f"RealMLP categorical feature missing: {column}")

    numeric_columns = [column for column in X_fit_raw.columns if column not in MLP_CATEGORICAL_COLUMNS]
    fit_num = X_fit_raw[numeric_columns].copy()
    valid_num = X_valid_raw[numeric_columns].copy()
    test_num = X_test_raw_fold[numeric_columns].copy()
    for frame in (fit_num, valid_num, test_num):
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    medians = fit_num.median(axis=0, skipna=True).fillna(0.0)
    missing_columns = [column for column in numeric_columns if fit_num[column].isna().any()]
    fit_missing = fit_num[missing_columns].isna().astype(np.float32)
    valid_missing = valid_num[missing_columns].isna().astype(np.float32)
    test_missing = test_num[missing_columns].isna().astype(np.float32)
    fit_num = fit_num.fillna(medians).astype(np.float32)
    valid_num = valid_num.fillna(medians).astype(np.float32)
    test_num = test_num.fillna(medians).astype(np.float32)
    if missing_columns:
        indicator_names = [f"{column}__IS_MISSING" for column in missing_columns]
        fit_missing.columns = indicator_names
        valid_missing.columns = indicator_names
        test_missing.columns = indicator_names
        fit_num = pd.concat([fit_num, fit_missing], axis=1)
        valid_num = pd.concat([valid_num, valid_missing], axis=1)
        test_num = pd.concat([test_num, test_missing], axis=1)

    fit_cat = pd.DataFrame(index=X_fit_raw.index)
    valid_cat = pd.DataFrame(index=X_valid_raw.index)
    test_cat = pd.DataFrame(index=X_test_raw_fold.index)
    category_maps = {}
    for column in MLP_CATEGORICAL_COLUMNS:
        fit_values = X_fit_raw[column].astype("string")
        valid_values = X_valid_raw[column].astype("string")
        test_values = X_test_raw_fold[column].astype("string")
        vocabulary = sorted(fit_values.dropna().unique().tolist())
        mapping = {value: index + 1 for index, value in enumerate(vocabulary)}
        unknown_code = len(mapping) + 1
        fit_codes = fit_values.map(mapping).fillna(0).astype(np.int32)
        valid_codes = valid_values.map(mapping).fillna(unknown_code).astype(np.int32)
        test_codes = test_values.map(mapping).fillna(unknown_code).astype(np.int32)
        valid_codes.loc[valid_values.isna()] = 0
        test_codes.loc[test_values.isna()] = 0
        fit_cat[column] = fit_codes
        valid_cat[column] = valid_codes
        test_cat[column] = test_codes
        category_maps[column] = {
            "vocabulary": vocabulary,
            "unknown_code": unknown_code,
            "missing_code": 0,
        }

    fit_out = pd.concat([fit_num, fit_cat], axis=1)
    valid_out = pd.concat([valid_num, valid_cat], axis=1)[fit_out.columns]
    test_out = pd.concat([test_num, test_cat], axis=1)[fit_out.columns]
    preprocessing = {
        "numeric_medians": {key: float(value) for key, value in medians.items()},
        "numeric_columns": numeric_columns,
        "numeric_missing_indicator_columns": missing_columns,
        "categorical_maps": category_maps,
        "categorical_columns": list(MLP_CATEGORICAL_COLUMNS),
        "output_columns": list(fit_out.columns),
    }
    return fit_out, valid_out, test_out, preprocessing


def prepare_gbdt_categorical_views(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test_fold: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    fit_view, valid_view, test_view = X_fit.copy(), X_valid.copy(), X_test_fold.copy()
    categories_by_column: dict[str, list[str]] = {}
    for column in categorical_columns:
        fit_values = X_fit[column].astype("string").fillna(MISSING_CATEGORY)
        valid_values = X_valid[column].astype("string").fillna(MISSING_CATEGORY)
        test_values = X_test_fold[column].astype("string").fillna(MISSING_CATEGORY)
        categories = list(pd.unique(fit_values))
        categories_by_column[column] = categories
        fit_view[column] = pd.Categorical(fit_values, categories=categories)
        valid_view[column] = pd.Categorical(valid_values, categories=categories)
        test_view[column] = pd.Categorical(test_values, categories=categories)
    return fit_view, valid_view, test_view, categories_by_column


# =============================================================================
# 11. BUILD / RESTORE FOLD-LOCAL FEATURE CACHES
# =============================================================================

TRAIN_IMPUTED_BY_FOLD: dict[int, dict[str, np.ndarray]] = {}
TEST_IMPUTED_BY_FOLD: dict[int, dict[str, np.ndarray]] = {}


def load_fold_feature_cache(fold_number: int) -> dict[str, Any]:
    fold_dir = CACHE_DIR / f"FOLD_{fold_number:02d}"
    cache_path = fold_dir / "FEATURE_CACHE.pkl"
    marker_path = fold_dir / "FEATURE_CACHE_COMPLETE.json"
    if not checkpoint_complete(marker_path, "feature_cache", CONFIG_HASH, [cache_path]):
        raise RuntimeError(f"Fold {fold_number}: feature cache is not complete for current configuration")
    payload = load_pickle(cache_path)
    if payload.get("config_sha256") != CONFIG_HASH or int(payload.get("fold", -1)) != fold_number:
        raise RuntimeError(f"Fold {fold_number}: feature-cache signature mismatch")
    return payload


def build_one_fold_feature_cache(
    fold_number: int,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
) -> None:
    fold_dir = CACHE_DIR / f"FOLD_{fold_number:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    cache_path = fold_dir / "FEATURE_CACHE.pkl"
    marker_path = fold_dir / "FEATURE_CACHE_COMPLETE.json"
    validation_diag_path = fold_dir / "VALIDATION_NEIGHBOR_DIAGNOSTICS.csv"
    test_diag_path = fold_dir / "TEST_NEIGHBOR_DIAGNOSTICS.csv"
    cluster_diag_path = fold_dir / "VALIDATION_CLUSTERS.csv"
    schema_path = fold_dir / "FEATURE_SCHEMA.json"
    preprocessing_path = PREPROCESSING_DIR / f"FOLD_{fold_number:02d}" / "PREPROCESSING_BUNDLE.pkl"
    preprocessing_path.parent.mkdir(parents=True, exist_ok=True)
    gmm_fold_dir = GMM_DIR / f"FOLD_{fold_number:02d}"
    gmm_model_path = gmm_fold_dir / "segmenter.pkl"
    gmm_metadata_path = gmm_fold_dir / "segmenter_metadata.json"
    gmm_status_path = gmm_fold_dir / "GMM_CACHE_STATUS.json"

    required = [
        cache_path,
        validation_diag_path,
        test_diag_path,
        cluster_diag_path,
        schema_path,
        preprocessing_path,
        gmm_model_path,
        gmm_metadata_path,
        gmm_status_path,
    ]
    if checkpoint_complete(marker_path, "feature_cache", CONFIG_HASH, required):
        restored = load_pickle(cache_path)
        if (
            restored.get("config_sha256") == CONFIG_HASH
            and np.array_equal(restored.get("fit_indices"), fit_indices)
            and np.array_equal(restored.get("valid_indices"), valid_indices)
        ):
            print(f"Fold {fold_number}: restored completed feature cache.", flush=True)
            return
        print(f"Fold {fold_number}: cache index/signature mismatch; rebuilding.", flush=True)

    print_section(f"FEATURE PIPELINE — FOLD {fold_number}/{N_SPLITS}")
    train_imputed = load_and_validate_training_imputations(fold_number)
    test_imputed = generate_test_imputations(fold_number, fit_indices)
    TRAIN_IMPUTED_BY_FOLD[fold_number] = train_imputed
    TEST_IMPUTED_BY_FOLD[fold_number] = test_imputed

    train_augmented = augment_with_ext_source(X_train_original, train_imputed)
    test_augmented = augment_with_ext_source(X_test_original, test_imputed)

    X_fit_raw = train_augmented.iloc[fit_indices].copy()
    X_valid_raw = train_augmented.iloc[valid_indices].copy()
    X_test_fold = test_augmented.copy()

    # The two requested features are constructed independently of TARGET.
  
    familly_fit = build_familly_dhouse_profile(X_fit_raw, X_fit_raw)
    familly_valid = build_familly_dhouse_profile(X_fit_raw, X_valid_raw)
    familly_test = build_familly_dhouse_profile(X_fit_raw, X_test_fold)
    X_fit_raw[FAMILLY_DHOUSE_FEATURE] = familly_fit
    X_valid_raw[FAMILLY_DHOUSE_FEATURE] = familly_valid
    X_test_fold[FAMILLY_DHOUSE_FEATURE] = familly_test

    # Restore or create one GMM segmenter using the persistent fingerprint.
    segmenter, gmm_metadata, gmm_cache_source = get_fold_gmm_segmenter(
        fold_number,
        fit_indices,
        X_train_original,
        train_ids,
    )
    cluster_fit = segmenter.transform(X_fit_raw)
    cluster_valid = segmenter.transform(X_valid_raw)
    cluster_test = segmenter.transform(X_test_fold)

    with threadpool_limits(limits=CPU_THREADS):
        local_fit, local_valid, local_test, neighbor_state, neighbor_diagnostics = (
            build_fold_local_neighbor_interaction(X_fit_raw, X_valid_raw, X_test_fold)
        )

    tree_statistics = fit_tree_feature_statistics(X_fit_raw)
    X_tree_fit = add_tree_engineered_features(X_fit_raw, tree_statistics)
    X_tree_valid = add_tree_engineered_features(X_valid_raw, tree_statistics)
    X_tree_test = add_tree_engineered_features(X_test_fold, tree_statistics)
    for frame, cluster, interaction in [
        (X_tree_fit, cluster_fit, local_fit),
        (X_tree_valid, cluster_valid, local_valid),
        (X_tree_test, cluster_test, local_test),
    ]:
        frame[LOAN_STRUCTURE_FEATURE] = cluster.astype("string")
        frame[NEW_INTERACTION_FEATURE] = np.asarray(interaction, dtype=np.float32)
        frame[FAMILLY_DHOUSE_FEATURE] = frame[FAMILLY_DHOUSE_FEATURE].astype("string")

    mlp_region_means, mlp_global_income_mean = fit_mlp_region_income_means(X_fit_raw)
    raw_augmented_columns = list(X_fit_raw.columns)
    X_mlp_fit_raw = build_mlp_feature_frame(
        X_fit_raw,
        mlp_region_means,
        mlp_global_income_mean,
        local_fit,
        cluster_fit,
        raw_augmented_columns,
    )
    X_mlp_valid_raw = build_mlp_feature_frame(
        X_valid_raw,
        mlp_region_means,
        mlp_global_income_mean,
        local_valid,
        cluster_valid,
        raw_augmented_columns,
    )
    X_mlp_test_raw = build_mlp_feature_frame(
        X_test_fold,
        mlp_region_means,
        mlp_global_income_mean,
        local_test,
        cluster_test,
        raw_augmented_columns,
    )
    X_mlp_fit, X_mlp_valid, X_mlp_test, mlp_preprocessing = encode_mlp_fold_inputs(
        X_mlp_fit_raw, X_mlp_valid_raw, X_mlp_test_raw
    )

    tree_categorical_columns = [
        column
        for column in X_tree_fit.columns
        if (
            pd.api.types.is_object_dtype(X_tree_fit[column].dtype)
            or pd.api.types.is_string_dtype(X_tree_fit[column].dtype)
            or isinstance(X_tree_fit[column].dtype, pd.CategoricalDtype)
        )
    ]
    tree_fit_lgb, tree_valid_lgb, tree_test_lgb, tree_categories = prepare_gbdt_categorical_views(
        X_tree_fit,
        X_tree_valid,
        X_tree_test,
        tree_categorical_columns,
    )

    if not (
        list(X_tree_fit.columns) == list(X_tree_valid.columns) == list(X_tree_test.columns)
    ):
        raise RuntimeError("Tree feature schema mismatch across fit/validation/test")
    if not (
        list(X_mlp_fit.columns) == list(X_mlp_valid.columns) == list(X_mlp_test.columns)
    ):
        raise RuntimeError("RealMLP feature schema mismatch across fit/validation/test")
    for required_feature in [FAMILLY_DHOUSE_FEATURE, LOAN_STRUCTURE_FEATURE]:
        if required_feature not in X_tree_fit.columns:
            raise RuntimeError(f"Tree view is missing selected feature {required_feature}")
    if "GOODS_PRICE_TO_ANNUITY" in X_tree_fit.columns or "GOODS_PRICE_TO_ANNUITY" in X_mlp_fit.columns:
        raise RuntimeError("GOODS_PRICE_TO_ANNUITY leaked into classifier features")

    cache_payload = {
        "fold": fold_number,
        "config_sha256": CONFIG_HASH,
        "fit_indices": np.asarray(fit_indices, dtype=np.int64),
        "valid_indices": np.asarray(valid_indices, dtype=np.int64),
        "y_fit": y[fit_indices].astype(np.int32),
        "y_valid": y[valid_indices].astype(np.int32),
        "tree_fit": X_tree_fit,
        "tree_valid": X_tree_valid,
        "tree_test": X_tree_test,
        "tree_fit_lgb": tree_fit_lgb,
        "tree_valid_lgb": tree_valid_lgb,
        "tree_test_lgb": tree_test_lgb,
        "tree_categorical_columns": tree_categorical_columns,
        "tree_categories": tree_categories,
        "mlp_fit": X_mlp_fit,
        "mlp_valid": X_mlp_valid,
        "mlp_test": X_mlp_test,
        "mlp_categorical_columns": list(MLP_CATEGORICAL_COLUMNS),
        "tree_feature_names": list(X_tree_fit.columns),
        "mlp_feature_names": list(X_mlp_fit.columns),
    }
    atomic_pickle(cache_path, cache_payload)

    validation_diag = neighbor_diagnostics["validation"].copy()
    validation_diag.insert(0, "FOLD", fold_number)
    validation_diag[ID_COL] = train_ids[valid_indices]
    test_diag = neighbor_diagnostics["test"].copy()
    test_diag.insert(0, "FOLD", fold_number)
    test_diag[ID_COL] = test_ids
    cluster_diag = pd.DataFrame(
        {
            ID_COL: train_ids[valid_indices],
            "FOLD": fold_number,
            TARGET_COL: y[valid_indices],
            LOAN_STRUCTURE_FEATURE: cluster_valid.astype(str).to_numpy(),
            FAMILLY_DHOUSE_FEATURE: familly_valid.astype(str).to_numpy(),
        }
    )
    atomic_csv(validation_diag_path, validation_diag)
    atomic_csv(test_diag_path, test_diag)
    atomic_csv(cluster_diag_path, cluster_diag)

    preprocessing_bundle = {
        "fold": fold_number,
        "config_sha256": CONFIG_HASH,
        "raw_feature_columns": feature_columns,
        "raw_categorical_columns": raw_categorical_columns,
        "ext_source_columns": EXT_SOURCE_COLUMNS,
        "ext_source_missing_columns": EXT_SOURCE_MISSING_COLUMNS,
        "tree_feature_names": list(X_tree_fit.columns),
        "tree_categorical_columns": tree_categorical_columns,
        "tree_category_vocabularies": tree_categories,
        "tree_statistics": tree_statistics,
        "realmlp_feature_names": list(X_mlp_fit.columns),
        "realmlp_categorical_columns": list(MLP_CATEGORICAL_COLUMNS),
        "realmlp_preprocessing": mlp_preprocessing,
        "realmlp_region_income_means": mlp_region_means,
        "realmlp_global_income_mean": mlp_global_income_mean,
        "neighbor_state": neighbor_state,
        "gmm_segmenter_path": str(GMM_DIR / f"FOLD_{fold_number:02d}" / "segmenter.pkl"),
        "gmm_cache_source": gmm_cache_source,
        "imputation_fold_dir": str(IMPUTATION_MODELS_DIR / f"FOLD_{fold_number:02d}"),
        "fit_indices": np.asarray(fit_indices, dtype=np.int64),
        "valid_indices": np.asarray(valid_indices, dtype=np.int64),
        "new_features": {
        
            FAMILLY_DHOUSE_FEATURE: {
                "source_columns": ["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"],
                "minimum_reference_support": FAMILLY_DHOUSE_MIN_COUNT,
                "fallback_order": [
                    "GENDER+FAMILY_STATUS+HOUSING_TYPE",
                    "GENDER+FAMILY_STATUS",
                    "GENDER+HOUSING_TYPE",
                    "GENDER",
                ],
                "fit_scope": "outer fold-training rows only; TARGET-free",
            },
        },
    }
    atomic_pickle(preprocessing_path, preprocessing_bundle)

    schema = {
        "fold": fold_number,
        "config_sha256": CONFIG_HASH,
        "tree_feature_names": list(X_tree_fit.columns),
        "tree_categorical_columns": tree_categorical_columns,
        "tree_engineered_features": TREE_ENGINEERED_FEATURES,
        "realmlp_feature_names_after_encoding": list(X_mlp_fit.columns),
        "realmlp_engineered_features": MLP_ENGINEERED_FEATURES,
        "realmlp_categorical_columns": MLP_CATEGORICAL_COLUMNS,
        "new_features": {
           
            FAMILLY_DHOUSE_FEATURE: {
                "minimum_support": FAMILLY_DHOUSE_MIN_COUNT,
                "fit_scope": "fold-training only",
                "target_used": False,
            },
        },
        "shared_features": EXT_SOURCE_COLUMNS + EXT_SOURCE_MISSING_COLUMNS + [
            NEW_INTERACTION_FEATURE,
            LOAN_STRUCTURE_FEATURE,
         
            FAMILLY_DHOUSE_FEATURE,
        ],
        "gmm_fit_rows": int(len(fit_indices)),
        "neighbor_reference_rows": int(len(fit_indices)),
        "neighbor_count": N_NEIGHBORS,
        "target_used_for_gmm": False,
    }
    atomic_json(schema_path, schema)
    mark_complete(marker_path, "feature_cache", CONFIG_HASH, required)

    print(
        f"Fold {fold_number}: cache saved | tree features={X_tree_fit.shape[1]} | "
        f"RealMLP inputs={X_mlp_fit.shape[1]} | GMM={gmm_cache_source}",
        flush=True,
    )

    del (
        train_imputed, test_imputed, train_augmented, test_augmented,
        X_fit_raw, X_valid_raw, X_test_fold,
        X_tree_fit, X_tree_valid, X_tree_test,
        tree_fit_lgb, tree_valid_lgb, tree_test_lgb,
        X_mlp_fit_raw, X_mlp_valid_raw, X_mlp_test_raw,
        X_mlp_fit, X_mlp_valid, X_mlp_test,
        cache_payload, neighbor_diagnostics, segmenter,
    )
    gc.collect()


# =============================================================================
# 12. MODEL FACTORIES AND PER-FOLD ARTIFACT MANAGEMENT
# =============================================================================

def catboost_params_for(model_name: str, fold_number: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        **FIXED_CATBOOST_PARAMS,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "devices": "0",
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "verbose": 200,
        "thread_count": CPU_THREADS,
    }
    if model_name == "catboost_symmetric":
        params["grow_policy"] = "SymmetricTree"
    elif model_name == "catboost_depthwise":
        params["grow_policy"] = "Depthwise"
        params["min_data_in_leaf"] = ASYMMETRIC_MIN_DATA_IN_LEAF
    elif model_name == "catboost_lossguide":
        params["grow_policy"] = "Lossguide"
        params["max_leaves"] = LOSSGUIDE_MAX_LEAVES
        params["min_data_in_leaf"] = ASYMMETRIC_MIN_DATA_IN_LEAF
    else:
        raise ValueError(f"Not a CatBoost model name: {model_name}")
    return params


def make_lightgbm_classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        **FIXED_LIGHTGBM_PARAMS,
        objective="binary",
        metric="auc",
        device_type="cpu",
        random_state=RANDOM_SEED,
        n_jobs=CPU_THREADS,
        subsample_freq=1,
        verbosity=-1,
    )


def make_xgboost_classifier() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **FIXED_XGBOOST_PARAMS,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        random_state=RANDOM_SEED,
        n_jobs=CPU_THREADS,
        verbosity=0,
    )


def model_fold_dir(model_name: str, fold_number: int) -> Path:
    path = MODEL_DIR / model_name.upper() / f"FOLD_{fold_number:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prediction_fold_dir(model_name: str, fold_number: int) -> Path:
    path = PREDICTION_DIR / model_name.upper() / f"FOLD_{fold_number:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_primary_path(model_name: str, fold_number: int) -> Path:
    directory = model_fold_dir(model_name, fold_number)
    if model_name.startswith("catboost_"):
        return directory / "catboost_model.cbm"
    if model_name == "lightgbm":
        return directory / "lightgbm_model.txt"
    if model_name == "xgboost":
        return directory / "xgboost_model.json"
    if model_name == "realmlp":
        return directory / "realmlp_model.pkl"
    raise ValueError(model_name)


def required_model_artifacts(model_name: str, fold_number: int) -> list[Path]:
    model_path = model_primary_path(model_name, fold_number)
    pred_dir = prediction_fold_dir(model_name, fold_number)
    required = [
        model_path,
        pred_dir / "PREDICTIONS.npz",
        pred_dir / "METRICS.json",
        model_fold_dir(model_name, fold_number) / "MODEL_METADATA.json",
    ]
    if model_name == "lightgbm":
        required.append(model_fold_dir(model_name, fold_number) / "lightgbm_sklearn_wrapper.pkl")
    elif model_name == "xgboost":
        required.append(model_fold_dir(model_name, fold_number) / "xgboost_sklearn_wrapper.pkl")
    if model_name in {"catboost_symmetric", "catboost_depthwise", "catboost_lossguide"}:
        required.append(IMPORTANCE_DIR / model_name.upper() / f"FOLD_{fold_number:02d}_IMPORTANCE.csv")
    elif model_name in {"lightgbm", "xgboost"}:
        required.append(IMPORTANCE_DIR / model_name.upper() / f"FOLD_{fold_number:02d}_IMPORTANCE.csv")
    return required


def load_model_fold_predictions(model_name: str, fold_number: int):
    pred_dir = prediction_fold_dir(model_name, fold_number)
    with np.load(pred_dir / "PREDICTIONS.npz", allow_pickle=False) as archive:
        valid_indices = np.asarray(archive["valid_indices"], dtype=np.int64)
        valid_prediction = np.asarray(archive["valid_prediction"], dtype=np.float64)
        test_prediction = np.asarray(archive["test_prediction"], dtype=np.float64)
    metrics = json.loads((pred_dir / "METRICS.json").read_text(encoding="utf-8"))
    return valid_indices, valid_prediction, test_prediction, metrics


def hyperparameters_for_metadata(model_name: str, fold_number: int) -> dict[str, Any]:
    if model_name.startswith("catboost_"):
        return catboost_params_for(model_name, fold_number)
    if model_name == "lightgbm":
        return {**FIXED_LIGHTGBM_PARAMS, "device_type": "cpu", "random_state": RANDOM_SEED}
    if model_name == "xgboost":
        return {**FIXED_XGBOOST_PARAMS, "tree_method": "hist", "device": "cuda", "random_state": RANDOM_SEED}
    if model_name == "realmlp":
        params = dict(REALMLP_PARAMS)
        params["random_state"] = RANDOM_SEED
        return params
    raise ValueError(model_name)


def save_model_fold(
    model_name: str,
    fold_number: int,
    model: Any,
    valid_indices: np.ndarray,
    valid_prediction: np.ndarray,
    test_prediction: np.ndarray,
    metrics: dict[str, float],
    feature_names: list[str],
    importance: pd.DataFrame | None,
    training_details: dict[str, Any] | None = None,
) -> None:
    model_dir = model_fold_dir(model_name, fold_number)
    pred_dir = prediction_fold_dir(model_name, fold_number)
    model_path = model_primary_path(model_name, fold_number)

    if model_name.startswith("catboost_"):
        atomic_model_save(model, model_path, "catboost")
        model_artifacts = [model_path]
    elif model_name == "lightgbm":
        atomic_model_save(model, model_path, "lightgbm")
        wrapper_path = model_dir / "lightgbm_sklearn_wrapper.pkl"
        atomic_model_save(model, wrapper_path, "pickle")
        model_artifacts = [model_path, wrapper_path]
    elif model_name == "xgboost":
        atomic_model_save(model, model_path, "xgboost")
        wrapper_path = model_dir / "xgboost_sklearn_wrapper.pkl"
        atomic_model_save(model, wrapper_path, "pickle")
        model_artifacts = [model_path, wrapper_path]
    elif model_name == "realmlp":
        atomic_model_save(model, model_path, "pickle")
        model_artifacts = [model_path]
    else:
        raise ValueError(model_name)

    prediction_path = pred_dir / "PREDICTIONS.npz"
    atomic_npz(
        prediction_path,
        valid_indices=np.asarray(valid_indices, dtype=np.int64),
        valid_prediction=np.asarray(valid_prediction, dtype=np.float64),
        test_prediction=np.asarray(test_prediction, dtype=np.float64),
    )
    metrics_path = pred_dir / "METRICS.json"
    atomic_json(
        metrics_path,
        {
            "model": model_name,
            "fold": fold_number,
            "validation_metrics": metrics,
            "valid_rows": int(len(valid_indices)),
            "test_rows": int(len(test_prediction)),
            "config_sha256": CONFIG_HASH,
            "training_details": training_details or {},
        },
    )
    model_hyperparameters = hyperparameters_for_metadata(model_name, fold_number)
    if (
        model_name.startswith("catboost_")
        and training_details
        and "final_tree_count" in training_details
    ):
        model_hyperparameters["iterations"] = int(training_details["final_tree_count"])

    metadata_path = model_dir / "MODEL_METADATA.json"
    atomic_json(
        metadata_path,
        {
            "model": model_name,
            "fold": fold_number,
            "model_path": str(model_path),
            "feature_names": list(feature_names),
            "validation_metrics": metrics,
            "config_sha256": CONFIG_HASH,
            "hyperparameters": model_hyperparameters,
            "training_details": training_details or {},
            "actual_tree_count": int(model.tree_count_) if model_name.startswith("catboost_") else None,
            "catboost_version": package_version("catboost"),
            "lightgbm_version": package_version("lightgbm"),
            "xgboost_version": xgb.__version__,
            "pytabkit_version": package_version("pytabkit"),
            "python": sys.version.split()[0],
        },
    )

    artifacts = model_artifacts + [prediction_path, metrics_path, metadata_path]
    if importance is not None:
        importance_path = IMPORTANCE_DIR / model_name.upper() / f"FOLD_{fold_number:02d}_IMPORTANCE.csv"
        atomic_csv(importance_path, importance)
        artifacts.append(importance_path)
    mark_complete(pred_dir / "MODEL_FOLD_COMPLETE.json", f"{model_name}_fold", CONFIG_HASH, artifacts)


def prepare_catboost_frame(frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in categorical_columns:
        output[column] = output[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
    return output


def get_catboost_inner_split(
    y_fit: np.ndarray,
    outer_fit_indices: np.ndarray,
    fold_number: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load or create the common stratified inner split for one outer fold."""
    y_fit = np.asarray(y_fit, dtype=np.int32)
    outer_fit_indices = np.asarray(outer_fit_indices, dtype=np.int64)
    if y_fit.ndim != 1 or outer_fit_indices.ndim != 1 or len(y_fit) != len(outer_fit_indices):
        raise ValueError("CatBoost inner split: y_fit and outer_fit_indices are misaligned")
    if len(np.unique(y_fit)) != 2:
        raise ValueError("CatBoost inner split requires both classes in outer training")

    random_state = int(RANDOM_SEED + INNER_VALIDATION_SEED_OFFSET + fold_number)
    split_path = CHECKPOINT_DIR / "CATBOOST_INNER_SPLITS" / f"FOLD_{fold_number:02d}.npz"
    split_path.parent.mkdir(parents=True, exist_ok=True)

    saved_train_indices = saved_valid_indices = None
    if split_path.is_file():
        with np.load(split_path, allow_pickle=False) as archive:
            train_positions = np.asarray(archive["inner_train_positions"], dtype=np.int64)
            valid_positions = np.asarray(archive["inner_valid_positions"], dtype=np.int64)
            saved_train_indices = np.asarray(archive["inner_train_original_indices"], dtype=np.int64)
            saved_valid_indices = np.asarray(archive["inner_valid_original_indices"], dtype=np.int64)
            saved_seed = int(np.asarray(archive["random_state"]).reshape(-1)[0])
        if saved_seed != random_state:
            raise RuntimeError(f"Fold {fold_number}: saved inner split has a different random state")
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=INNER_VALIDATION_FRACTION,
            random_state=random_state,
        )
        train_positions, valid_positions = next(
            splitter.split(np.zeros(len(y_fit), dtype=np.int8), y_fit)
        )
        train_positions = np.asarray(train_positions, dtype=np.int64)
        valid_positions = np.asarray(valid_positions, dtype=np.int64)
        atomic_npz(
            split_path,
            inner_train_positions=train_positions,
            inner_valid_positions=valid_positions,
            inner_train_original_indices=outer_fit_indices[train_positions],
            inner_valid_original_indices=outer_fit_indices[valid_positions],
            random_state=np.asarray([random_state], dtype=np.int64),
        )

    # Validate the persisted positions before using them for .iloc selection.
    if (
        train_positions.ndim != 1
        or valid_positions.ndim != 1
        or len(train_positions) == 0
        or len(valid_positions) == 0
    ):
        raise RuntimeError(f"Fold {fold_number}: inner split positions are empty or malformed")
    if (
        train_positions.min() < 0
        or valid_positions.min() < 0
        or train_positions.max() >= len(y_fit)
        or valid_positions.max() >= len(y_fit)
    ):
        raise RuntimeError(f"Fold {fold_number}: inner split positions are out of range")
    if len(np.unique(train_positions)) != len(train_positions):
        raise RuntimeError(f"Fold {fold_number}: duplicate inner-training positions")
    if len(np.unique(valid_positions)) != len(valid_positions):
        raise RuntimeError(f"Fold {fold_number}: duplicate inner-validation positions")
    if saved_train_indices is not None:
        if not np.array_equal(saved_train_indices, outer_fit_indices[train_positions]):
            raise RuntimeError(f"Fold {fold_number}: saved inner-training indices do not match this run")
        if not np.array_equal(saved_valid_indices, outer_fit_indices[valid_positions]):
            raise RuntimeError(f"Fold {fold_number}: saved inner-validation indices do not match this run")
    if np.intersect1d(train_positions, valid_positions).size:
        raise RuntimeError(f"Fold {fold_number}: inner train/validation positions overlap")
    if len(train_positions) + len(valid_positions) != len(y_fit):
        raise RuntimeError(f"Fold {fold_number}: inner split does not cover outer training rows")
    if np.unique(y_fit[train_positions]).size != 2 or np.unique(y_fit[valid_positions]).size != 2:
        raise RuntimeError(f"Fold {fold_number}: an inner partition is missing a class")
    return train_positions, valid_positions, random_state


def run_one_model_fold(
    model_name: str,
    fold_number: int,
    cache: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    pred_dir = prediction_fold_dir(model_name, fold_number)
    marker = pred_dir / "MODEL_FOLD_COMPLETE.json"
    required = required_model_artifacts(model_name, fold_number)
    if checkpoint_complete(marker, f"{model_name}_fold", CONFIG_HASH, required):
        print(f"{model_name.upper()} fold {fold_number}: restoring saved model/predictions.", flush=True)
        return load_model_fold_predictions(model_name, fold_number)

    y_fit = cache["y_fit"]
    y_valid = cache["y_valid"]
    importance = None
    training_details: dict[str, Any] = {}

    if model_name.startswith("catboost_"):
        X_fit = prepare_catboost_frame(cache["tree_fit"], cache["tree_categorical_columns"])
        X_valid = prepare_catboost_frame(cache["tree_valid"], cache["tree_categorical_columns"])
        X_test_fold = prepare_catboost_frame(cache["tree_test"], cache["tree_categorical_columns"])

        # All three growth policies use the identical inner split for this outer fold.
        # The outer validation partition is not passed to the iteration-selection fit.
        inner_train_positions, inner_valid_positions, inner_random_state = get_catboost_inner_split(
            y_fit,
            cache["fit_indices"],
            fold_number,
        )
        X_inner_fit = X_fit.iloc[inner_train_positions]
        y_inner_fit = y_fit[inner_train_positions]
        X_inner_valid = X_fit.iloc[inner_valid_positions]
        y_inner_valid = y_fit[inner_valid_positions]

        pilot_params = catboost_params_for(model_name, fold_number)
        pilot_model = CatBoostClassifier(**pilot_params)
        with threadpool_limits(limits=CPU_THREADS):
            pilot_model.fit(
                X_inner_fit,
                y_inner_fit,
                eval_set=(X_inner_valid, y_inner_valid),
                cat_features=cache["tree_categorical_columns"],
                early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS,
                use_best_model=True,
                verbose=200,
            )

        best_iteration = int(pilot_model.get_best_iteration())
        selected_tree_count = int(pilot_model.tree_count_)
        if selected_tree_count < 1 or best_iteration < 0:
            raise RuntimeError(
                f"{model_name} fold {fold_number}: inner selection did not return a valid best iteration"
            )
        if best_iteration + 1 != selected_tree_count:
            raise RuntimeError(
                f"{model_name} fold {fold_number}: best_iteration={best_iteration} "
                f"but pilot tree_count_={selected_tree_count}"
            )
        if selected_tree_count > int(FIXED_CATBOOST_PARAMS["iterations"]):
            raise RuntimeError(
                f"{model_name} fold {fold_number}: selected tree count exceeds the configured maximum"
            )

        best_validation_auc = None
        best_scores = pilot_model.get_best_score()
        validation_score_groups = [
            score_group
            for split_name, score_group in best_scores.items()
            if str(split_name).strip().lower() not in {"learn", "training"}
            and isinstance(score_group, dict)
        ]
        for validation_scores in validation_score_groups:
            for metric_name, metric_value in validation_scores.items():
                if str(metric_name).upper().startswith("AUC"):
                    best_validation_auc = float(metric_value)
                    break
            if best_validation_auc is not None:
                break
        if best_validation_auc is None or not np.isfinite(best_validation_auc):
            raise RuntimeError(
                f"{model_name} fold {fold_number}: could not retrieve finite inner-validation AUC"
            )

        pilot_tree_count = int(pilot_model.tree_count_)
        del pilot_model, X_inner_fit, X_inner_valid, y_inner_fit, y_inner_valid
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Refit a NEW model on the complete outer-training fold with the selected
        # number of trees. No outer-validation data are used in this fit.
        final_params = catboost_params_for(model_name, fold_number)
        final_params["iterations"] = selected_tree_count
        model = CatBoostClassifier(**final_params)
        with threadpool_limits(limits=CPU_THREADS):
            model.fit(
                X_fit,
                y_fit,
                cat_features=cache["tree_categorical_columns"],
                verbose=200,
            )

        final_tree_count = int(model.tree_count_)
        if final_tree_count != selected_tree_count:
            raise RuntimeError(
                f"{model_name} fold {fold_number}: requested {selected_tree_count} trees "
                f"but final refit produced {final_tree_count}"
            )
        training_details = {
            "protocol": "inner_stratified_validation_then_full_outer_fold_refit",
            "inner_splitter": "StratifiedShuffleSplit",
            "inner_validation_fraction": float(INNER_VALIDATION_FRACTION),
            "inner_random_state": int(inner_random_state),
            "inner_training_rows": int(len(inner_train_positions)),
            "inner_validation_rows": int(len(inner_valid_positions)),
            "inner_best_iteration_zero_based": int(best_iteration),
            "selected_tree_count": int(selected_tree_count),
            "pilot_tree_count_after_use_best_model": int(pilot_tree_count),
            "inner_best_validation_auc": float(best_validation_auc),
            "final_refit_training_rows": int(len(y_fit)),
            "final_tree_count": int(final_tree_count),
            "early_stopping_rounds": int(CATBOOST_EARLY_STOPPING_ROUNDS),
            "max_iterations": int(FIXED_CATBOOST_PARAMS["iterations"]),
            "selection_metric": "AUC",
            "outer_validation_used_for_iteration_selection": False,
            "refit_on_full_outer_training_fold": True,
        }

        valid_prediction = np.asarray(model.predict_proba(X_valid)[:, 1], dtype=np.float64)
        test_prediction = np.asarray(model.predict_proba(X_test_fold)[:, 1], dtype=np.float64)
        importance = pd.DataFrame(
            {
                "FEATURE": cache["tree_feature_names"],
                "IMPORTANCE": model.get_feature_importance(type="PredictionValuesChange"),
            }
        ).sort_values("IMPORTANCE", ascending=False)
        feature_names = cache["tree_feature_names"]

    elif model_name == "lightgbm":
        X_fit = cache["tree_fit_lgb"]
        X_valid = cache["tree_valid_lgb"]
        X_test_fold = cache["tree_test_lgb"]
        model = make_lightgbm_classifier()
        with threadpool_limits(limits=CPU_THREADS):
            model.fit(X_fit, y_fit, categorical_feature=cache["tree_categorical_columns"])
        valid_prediction = np.asarray(model.predict_proba(X_valid)[:, 1], dtype=np.float64)
        test_prediction = np.asarray(model.predict_proba(X_test_fold)[:, 1], dtype=np.float64)
        importance = pd.DataFrame(
            {
                "FEATURE": cache["tree_feature_names"],
                "IMPORTANCE_GAIN": model.booster_.feature_importance(importance_type="gain"),
                "IMPORTANCE_SPLIT": model.booster_.feature_importance(importance_type="split"),
            }
        ).sort_values("IMPORTANCE_GAIN", ascending=False)
        feature_names = cache["tree_feature_names"]

    elif model_name == "xgboost":
        X_fit = cache["tree_fit_lgb"]
        X_valid = cache["tree_valid_lgb"]
        X_test_fold = cache["tree_test_lgb"]
        model = make_xgboost_classifier()
        with threadpool_limits(limits=CPU_THREADS):
            model.fit(X_fit, y_fit, verbose=False)
        valid_prediction = np.asarray(model.predict_proba(X_valid)[:, 1], dtype=np.float64)
        test_prediction = np.asarray(model.predict_proba(X_test_fold)[:, 1], dtype=np.float64)
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        weight = booster.get_score(importance_type="weight")
        feature_names = cache["tree_feature_names"]
        importance = pd.DataFrame(
            {
                "FEATURE": feature_names,
                "IMPORTANCE_GAIN": [gain.get(name, 0.0) for name in feature_names],
                "IMPORTANCE_WEIGHT": [weight.get(name, 0.0) for name in feature_names],
            }
        ).sort_values("IMPORTANCE_GAIN", ascending=False)
        del booster

    elif model_name == "realmlp":
        X_fit = cache["mlp_fit"]
        X_valid = cache["mlp_valid"]
        X_test_fold = cache["mlp_test"]
        params = dict(REALMLP_PARAMS)
        params["random_state"] = RANDOM_SEED
        model = RealMLP_TD_Classifier(**params)
        with threadpool_limits(limits=CPU_THREADS):
            model.fit(
                X_fit,
                y_fit,
                X_val=X_valid,
                y_val=y_valid,
                cat_col_names=cache["mlp_categorical_columns"],
            )
        valid_prediction = positive_class_probability(model, X_valid)
        test_prediction = positive_class_probability(model, X_test_fold)
        feature_names = cache["mlp_feature_names"]

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    check_probabilities(f"{model_name} fold {fold_number} validation", valid_prediction)
    check_probabilities(f"{model_name} fold {fold_number} test", test_prediction)
    if len(valid_prediction) != len(cache["valid_indices"]) or len(test_prediction) != n_test:
        raise RuntimeError(f"{model_name} fold {fold_number}: prediction length mismatch")

    metrics = calculate_metrics(y_valid, valid_prediction)
    save_model_fold(
        model_name,
        fold_number,
        model,
        cache["valid_indices"],
        valid_prediction,
        test_prediction,
        metrics,
        feature_names,
        importance,
        training_details=training_details,
    )
    print(
        f"{model_name.upper()} fold {fold_number}: "
        f"AUC={metrics['roc_auc']:.7f} | AP={metrics['average_precision']:.7f}",
        flush=True,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        np.asarray(cache["valid_indices"], dtype=np.int64),
        valid_prediction,
        test_prediction,
        {"validation_metrics": metrics, "training_details": training_details},
    )


# =============================================================================
# 13. INPUT / IMPUTATION SIGNATURES AND EXPERIMENT CONFIGURATION
# =============================================================================

print_section("CALCULATING INPUT AND IMPUTATION SIGNATURES")
TRAIN_SIGNATURE = file_signature(TRAIN_FILE)
TEST_SIGNATURE = file_signature(TEST_FILE)
SAMPLE_SIGNATURE = file_signature(SAMPLE_SUB_FILE) if SAMPLE_SUB_FILE.is_file() else None
IMPUTATION_SIGNATURES: dict[str, Any] = {}
for fold_number in range(1, N_SPLITS + 1):
    IMPUTATION_SIGNATURES[f"fold_{fold_number}_values"] = file_signature(IMPUTATION_VALUES_PATHS[fold_number])
    IMPUTATION_SIGNATURES[f"fold_{fold_number}_indices"] = file_signature(IMPUTATION_INDEX_PATHS[fold_number])
    for source in EXT_SOURCE_COLUMNS:
        IMPUTATION_SIGNATURES[f"fold_{fold_number}_{source}_model"] = file_signature(
            IMPUTER_MODEL_PATHS[(fold_number, source)]
        )

config_payload = {
  
    "pipeline_version": "2026-11-18-six-model-gpu-inner-validation-refit-v3",
    "train_file": TRAIN_SIGNATURE,
    "test_file": TEST_SIGNATURE,
    "sample_submission_file": SAMPLE_SIGNATURE,
    "imputation_experiment_dir": str(IMPUTATION_EXPERIMENT_DIR),
    "imputation_artifacts": IMPUTATION_SIGNATURES,
    "train_shape": list(train.shape),
    "test_shape": list(test.shape),
    "raw_feature_columns": feature_columns,
    "raw_categorical_columns": raw_categorical_columns,
    "folds": {"method": "StratifiedKFold", "n_splits": N_SPLITS, "shuffle": True, "random_state": RANDOM_SEED},
    "devices": {"realmlp": "cuda:0", "catboost": "GPU 0", "lightgbm": "CPU", "xgboost": "cuda:0"},
    "model_names": MODEL_NAMES,
    "baseline_model_names": BASELINE_MODEL_NAMES,
    "baseline_ensemble_weights": BASELINE_ENSEMBLE_WEIGHTS,
    "catboost_common_params": FIXED_CATBOOST_PARAMS,
    "catboost_iteration_selection": {
        "method": "inner_stratified_validation_then_full_outer_fold_refit",
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "inner_splitter": "StratifiedShuffleSplit",
        "inner_random_state_rule": "RANDOM_SEED + INNER_VALIDATION_SEED_OFFSET + fold_number",
        "max_iterations": FIXED_CATBOOST_PARAMS["iterations"],
        "early_stopping_rounds": CATBOOST_EARLY_STOPPING_ROUNDS,
        "selection_metric": "AUC",
        "use_best_model_for_selection_fit": True,
        "refit_on_full_outer_training_fold": True,
        "outer_validation_used_for_iteration_selection": False,
        "same_inner_split_for_all_three_catboost_policies": True,
        "same_catboost_random_seed_across_policies": True,
    },
    "catboost_architectures": {
        "symmetric": {"grow_policy": "SymmetricTree"},
        "depthwise": {
            "grow_policy": "Depthwise",
            "min_data_in_leaf": ASYMMETRIC_MIN_DATA_IN_LEAF,
        },
        "lossguide": {
            "grow_policy": "Lossguide",
            "max_leaves": LOSSGUIDE_MAX_LEAVES,
            "min_data_in_leaf": ASYMMETRIC_MIN_DATA_IN_LEAF,
        },
    },
    "lightgbm_params": FIXED_LIGHTGBM_PARAMS,
    "xgboost_params": FIXED_XGBOOST_PARAMS,
    "realmlp_params": REALMLP_PARAMS,
    "new_features": {

        FAMILLY_DHOUSE_FEATURE: {
            "source_columns": ["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"],
            "minimum_reference_support": FAMILLY_DHOUSE_MIN_COUNT,
            "fallback_order": [
                "GENDER+FAMILY_STATUS+HOUSING_TYPE",
                "GENDER+FAMILY_STATUS",
                "GENDER+HOUSING_TYPE",
                "GENDER",
            ],
            "target_used": False,
            "fit_scope": "outer fold-training rows only",
        },
    },
    "saved_imputation": {
        "method": "reuse saved fold-local IMPUTED_VALUES.npz and fold-specific CatBoost imputers",
        "imputer_input_feature_count": EXPECTED_SAVED_IMPUTER_FEATURE_COUNT,
        "GOODS_PRICE_TO_ANNUITY": "saved-imputer input only; excluded from classifier views",
    },
    "gmm_cache": {
        "directory": str(GMM_CACHE_DIR),
        "schema_version": GMM_CACHE_SCHEMA_VERSION,
        "input_columns": GMM_CACHE_INPUT_COLUMNS,
        "max_components": GMM_MAX_COMPONENTS,
        "min_component_rows": GMM_MIN_COMPONENT_ROWS,
        "stability_repeats": GMM_STABILITY_REPEATS,
    },
    "neighbor_count": N_NEIGHBORS,
    "library_versions": {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": package_version("scikit-learn"),
        "torch": torch.__version__,
        "catboost": package_version("catboost"),
        "lightgbm": package_version("lightgbm"),
        "xgboost": xgb.__version__,
        "pytabkit": package_version("pytabkit"),
    },
}
CONFIG_HASH = sha256_json(config_payload)
RUN_CONFIG_PATH = OUTPUT_DIR / "RUN_CONFIG.json"
if RUN_CONFIG_PATH.is_file():
    old_config = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
    if old_config.get("config_sha256") != CONFIG_HASH:
        previous = OUTPUT_DIR / f"RUN_CONFIG_PREVIOUS_{old_config.get('config_sha256', 'unknown')[:12]}.json"
        if not previous.exists():
            atomic_json(previous, old_config)
        raise RuntimeError(
            "Existing output directory contains a different run configuration. "
            f"To prevent mixing incompatible artifacts, use a fresh OUTPUT_DIR. Current hash={CONFIG_HASH}"
        )
else:
    atomic_json(
        RUN_CONFIG_PATH,
        {
            "config_sha256": CONFIG_HASH,
            "configuration": config_payload,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


# =============================================================================
# 14. BUILD / RESTORE ALL FIVE FEATURE FOLDS
# =============================================================================

print_section("BUILDING / RESTORING FIVE FOLD FEATURE CACHES")
for fold_number, (fit_indices, valid_indices) in enumerate(fold_splits, start=1):
    build_one_fold_feature_cache(fold_number, fit_indices, valid_indices)
    gc.collect()

# Re-load per-fold imputation values into the in-memory namespace only if useful
# to the current session. The model-training stage consumes cached features.

# Rebuild global feature/GMM diagnostics from saved fold artifacts.
feature_schema_by_fold = {}
validation_neighbor_diagnostics = []
test_neighbor_diagnostics = []
validation_clusters = []
gmm_fold_metadata = []
gmm_status_rows = []
for fold_number in range(1, N_SPLITS + 1):
    fold_dir = CACHE_DIR / f"FOLD_{fold_number:02d}"
    feature_schema_by_fold[str(fold_number)] = json.loads(
        (fold_dir / "FEATURE_SCHEMA.json").read_text(encoding="utf-8")
    )
    validation_neighbor_diagnostics.append(pd.read_csv(fold_dir / "VALIDATION_NEIGHBOR_DIAGNOSTICS.csv"))
    test_neighbor_diagnostics.append(pd.read_csv(fold_dir / "TEST_NEIGHBOR_DIAGNOSTICS.csv"))
    validation_clusters.append(pd.read_csv(fold_dir / "VALIDATION_CLUSTERS.csv"))
    gmm_fold_dir = GMM_DIR / f"FOLD_{fold_number:02d}"
    gmm_fold_metadata.append(json.loads((gmm_fold_dir / "segmenter_metadata.json").read_text(encoding="utf-8")))
    gmm_status_rows.append(json.loads((gmm_fold_dir / "GMM_CACHE_STATUS.json").read_text(encoding="utf-8")))

atomic_json(OUTPUT_DIR / "FEATURE_SCHEMA_BY_FOLD.json", feature_schema_by_fold)
atomic_csv(
    OUTPUT_DIR / "VALIDATION_NEIGHBOR_DIAGNOSTICS_BY_FOLD.csv",
    pd.concat(validation_neighbor_diagnostics, ignore_index=True),
)
atomic_csv(
    OUTPUT_DIR / "TEST_NEIGHBOR_DIAGNOSTICS_BY_FOLD.csv",
    pd.concat(test_neighbor_diagnostics, ignore_index=True),
)
atomic_csv(
    OUTPUT_DIR / "VALIDATION_LOAN_STRUCTURE_CLUSTERS_BY_FOLD.csv",
    pd.concat(validation_clusters, ignore_index=True),
)
atomic_csv(GMM_DIR / "GMM_CACHE_STATUS_BY_FOLD.csv", pd.DataFrame(gmm_status_rows))
atomic_json(
    GMM_DIR / "GMM_CACHE_MANIFEST.json",
    {
        "cache_directory": str(GMM_CACHE_DIR),
        "cache_schema_version": GMM_CACHE_SCHEMA_VERSION,
        "cache_input_columns": GMM_CACHE_INPUT_COLUMNS,
        "target_used_for_gmm_fitting": False,
        "folds": gmm_status_rows,
    },
)
atomic_json(
    GMM_DIR / "HYBRID_GMM_SEGMENTATION_BY_FOLD.json",
    {
        "gmm_max_components": GMM_MAX_COMPONENTS,
        "gmm_min_component_rows": GMM_MIN_COMPONENT_ROWS,
        "gmm_stability_repeats": GMM_STABILITY_REPEATS,
        "target_used": False,
        "folds": gmm_fold_metadata,
    },
)


# =============================================================================
# 15. TRAIN SIX CLASSIFIERS — FIVE FOLDS EACH
# =============================================================================

print_section("TRAINING SIX CLASSIFIERS — 30 FINAL MODELS + 15 CATBOOST SELECTION FITS")
for fold_number in range(1, N_SPLITS + 1):
    cache = load_fold_feature_cache(fold_number)
    for model_name in MODEL_NAMES:
        started = time.perf_counter()
        run_one_model_fold(model_name, fold_number, cache)
        elapsed = (time.perf_counter() - started) / 60.0
        print(f"{model_name.upper()} fold {fold_number}: elapsed={elapsed:.2f} min", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del cache
    gc.collect()


# =============================================================================
# 16. RECONSTRUCT OOF / TEST PREDICTIONS AND METRICS
# =============================================================================

print_section("RECONSTRUCTING SIX-MODEL OOF AND TEST PREDICTIONS")
oof_predictions: dict[str, np.ndarray] = {}
test_predictions_by_fold: dict[str, np.ndarray] = {}
fold_metric_rows = []

for model_name in MODEL_NAMES:
    oof = np.full(n_train, np.nan, dtype=np.float64)
    test_by_fold = np.full((N_SPLITS, n_test), np.nan, dtype=np.float64)
    for fold_number in range(1, N_SPLITS + 1):
        valid_indices, valid_prediction, test_prediction, metrics_payload = load_model_fold_predictions(
            model_name, fold_number
        )
        if len(valid_indices) != len(valid_prediction) or len(test_prediction) != n_test:
            raise RuntimeError(f"Invalid saved prediction dimensions: {model_name}, fold {fold_number}")
        if np.isfinite(oof[valid_indices]).any():
            raise RuntimeError(f"Duplicate OOF assignments for {model_name}, fold {fold_number}")
        oof[valid_indices] = valid_prediction
        test_by_fold[fold_number - 1] = test_prediction
        fold_metric_rows.append(
            {
                "MODEL": model_name,
                "FOLD": fold_number,
                **metrics_payload["validation_metrics"],
            }
        )
    if not np.isfinite(oof).all() or not np.isfinite(test_by_fold).all():
        raise RuntimeError(f"Incomplete OOF or TEST predictions for {model_name}")
    check_probabilities(f"{model_name} OOF", oof)
    for fold_index in range(N_SPLITS):
        check_probabilities(f"{model_name} TEST fold {fold_index + 1}", test_by_fold[fold_index])
    oof_predictions[model_name] = oof
    test_predictions_by_fold[model_name] = test_by_fold

fold_metrics_df = pd.DataFrame(fold_metric_rows)
atomic_csv(ENSEMBLE_DIR / "MODEL_FOLD_METRICS.csv", fold_metrics_df)

catboost_selection_rows = []
for catboost_name in ["catboost_symmetric", "catboost_depthwise", "catboost_lossguide"]:
    for fold_number in range(1, N_SPLITS + 1):
        _, _, _, saved_metrics = load_model_fold_predictions(catboost_name, fold_number)
        details = saved_metrics.get("training_details", {})
        catboost_selection_rows.append(
            {
                "MODEL": catboost_name,
                "FOLD": fold_number,
                "INNER_BEST_ITERATION_ZERO_BASED": details.get("inner_best_iteration_zero_based"),
                "SELECTED_TREE_COUNT": details.get("selected_tree_count"),
                "FINAL_TREE_COUNT": details.get("final_tree_count"),
                "INNER_BEST_VALIDATION_AUC": details.get("inner_best_validation_auc"),
                "INNER_TRAINING_ROWS": details.get("inner_training_rows"),
                "INNER_VALIDATION_ROWS": details.get("inner_validation_rows"),
                "FINAL_REFIT_TRAINING_ROWS": details.get("final_refit_training_rows"),
                "INNER_RANDOM_STATE": details.get("inner_random_state"),
                "OUTER_VALIDATION_USED_FOR_SELECTION": details.get("outer_validation_used_for_iteration_selection"),
            }
        )
catboost_selection_summary_path = ENSEMBLE_DIR / "CATBOOST_INNER_SELECTION_SUMMARY.csv"
atomic_csv(catboost_selection_summary_path, pd.DataFrame(catboost_selection_rows))

model_test_mean = {
    model_name: test_predictions_by_fold[model_name].mean(axis=0)
    for model_name in MODEL_NAMES
}

# The four-model ensemble is retained strictly as the historical baseline.
baseline_ensemble_oof = sum(
    BASELINE_ENSEMBLE_WEIGHTS[name] * oof_predictions[name]
    for name in BASELINE_MODEL_NAMES
)
baseline_ensemble_test = sum(
    BASELINE_ENSEMBLE_WEIGHTS[name] * model_test_mean[name]
    for name in BASELINE_MODEL_NAMES
)
check_probabilities("Four-model baseline OOF", baseline_ensemble_oof)
check_probabilities("Four-model baseline TEST", baseline_ensemble_test)

fold_summary_rows = []
pair_counts = []
for fold_number, (_, valid_indices) in enumerate(fold_splits, start=1):
    y_valid = y[valid_indices]
    pair_counts.append(int(np.sum(y_valid == 1) * np.sum(y_valid == 0)))
    row: dict[str, Any] = {
        "FOLD": fold_number,
        "N_VALID": int(len(valid_indices)),
        "N_POSITIVE": int(y_valid.sum()),
        "N_NEGATIVE": int((y_valid == 0).sum()),
    }
    for model_name in MODEL_NAMES:
        metrics = calculate_metrics(y_valid, oof_predictions[model_name][valid_indices])
        for metric_name, value in metrics.items():
            row[f"{model_name.upper()}_{metric_name.upper()}"] = value
    baseline_metrics = calculate_metrics(y_valid, baseline_ensemble_oof[valid_indices])
    for metric_name, value in baseline_metrics.items():
        row[f"FOUR_MODEL_BASELINE_{metric_name.upper()}"] = value
    fold_summary_rows.append(row)

fold_summary_df = pd.DataFrame(fold_summary_rows)
atomic_csv(ENSEMBLE_DIR / "SIX_MODEL_FIVE_FOLD_METRICS_WITH_FOUR_MODEL_BASELINE.csv", fold_summary_df)

baseline_fold_auc = fold_summary_df["FOUR_MODEL_BASELINE_ROC_AUC"].to_numpy(dtype=np.float64)
pair_weighted_baseline_auc = float(
    np.average(baseline_fold_auc, weights=np.asarray(pair_counts, dtype=np.float64))
)
baseline_pooled_metrics = calculate_metrics(y, baseline_ensemble_oof)
model_oof_metrics = {
    name: calculate_metrics(y, oof_predictions[name])
    for name in MODEL_NAMES
}

# OOF predictions: six individual models + reference four-model ensemble.
oof_frame = pd.DataFrame({ID_COL: train_ids, TARGET_COL: y})
for name in MODEL_NAMES:
    oof_frame[f"{name.upper()}_OOF_PROBABILITY"] = oof_predictions[name]
oof_frame["FOUR_MODEL_BASELINE_EQUAL_WEIGHT_OOF"] = baseline_ensemble_oof
atomic_csv(ENSEMBLE_DIR / "OOF_PREDICTIONS_ALL_SIX_MODELS.csv", oof_frame)

# TEST predictions: preserve each fold's predictions and the five-fold mean.
test_prediction_frame = pd.DataFrame({ID_COL: test_ids})
for name in MODEL_NAMES:
    for fold_index in range(N_SPLITS):
        test_prediction_frame[f"{name.upper()}_FOLD_{fold_index + 1}"] = test_predictions_by_fold[name][fold_index]
    test_prediction_frame[f"{name.upper()}_MEAN_5FOLD"] = model_test_mean[name]
test_prediction_frame["FOUR_MODEL_BASELINE_EQUAL_WEIGHT"] = baseline_ensemble_test
atomic_csv(ENSEMBLE_DIR / "TEST_PREDICTIONS_ALL_SIX_MODELS_AND_FOLDS.csv", test_prediction_frame)

# Individual candidate submissions.
sample_template = pd.DataFrame({ID_COL: test_ids, TARGET_COL: np.nan})

def write_submission(probabilities: np.ndarray, path: Path, label: str) -> None:
    check_probabilities(label, probabilities)
    if len(probabilities) != n_test:
        raise ValueError(f"{label}: test prediction count does not match test.csv")
    output = sample_template.copy()
    output[TARGET_COL] = probabilities
    atomic_csv(path, output)

for name in MODEL_NAMES:
    write_submission(model_test_mean[name], ENSEMBLE_DIR / f"SUBMISSION_{name.upper()}_5FOLD.csv", name)
write_submission(
    baseline_ensemble_test,
    ENSEMBLE_DIR / "SUBMISSION_FOUR_MODEL_BASELINE_EQUAL_WEIGHT.csv",
    "four-model baseline",
)

# Preserve an individual OOF summary for later advanced blending.
summary = {
    "experiment": "HOME_CREDIT_SIX_MODEL_GPU_REUSED_IMPUTATION_PLUS_TWO_FEATURES",
    "status": "COMPLETE",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "config_sha256": CONFIG_HASH,
    "input_files": {
        "train": str(TRAIN_FILE),
        "train_sha256": TRAIN_SIGNATURE["sha256"],
        "test": str(TEST_FILE),
        "test_sha256": TEST_SIGNATURE["sha256"],
    },
    "imputation": {
        "retrained": False,
        "reused_saved_training_values": True,
        "reused_saved_fold_imputers_for_test": True,
        "fold_count": N_SPLITS,
        "imputation_models_reused": len(EXT_SOURCE_COLUMNS) * N_SPLITS,
        "goods_price_to_annuity_classifier_feature": False,
    },
    "rows": {
        "train": n_train,
        "test": n_test,
        "positive_train": int(y.sum()),
        "positive_rate": float(y.mean()),
    },
    "folds": {
        "method": "StratifiedKFold",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_SEED,
    },
    "devices": {
        "realmlp": "cuda:0",
        "catboost_symmetric": "GPU 0",
        "catboost_depthwise": "GPU 0",
        "catboost_lossguide": "GPU 0",
        "lightgbm": "CPU",
        "xgboost": "cuda:0",
    },
    "classification_models": len(MODEL_NAMES) * N_SPLITS,
    "catboost_iteration_selection": config_payload["catboost_iteration_selection"],
    "catboost_inner_selection_fit_count": 3 * N_SPLITS,
    "catboost_final_refit_count": 3 * N_SPLITS,
    "total_classifier_fit_calls_including_catboost_selection": len(MODEL_NAMES) * N_SPLITS + 3 * N_SPLITS,
    "model_names": MODEL_NAMES,
    "new_features": {
     
        FAMILLY_DHOUSE_FEATURE: {
            "minimum_support": FAMILLY_DHOUSE_MIN_COUNT,
            "source_columns": ["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"],
            "target_used": False,
        },
    },
    "gmm_cache_directory": str(GMM_CACHE_DIR),
    "gmm_fold_sources": [row.get("cache_source") for row in gmm_status_rows],
    "individual_pooled_oof_metrics": model_oof_metrics,
    "four_model_baseline": {
        "weights": BASELINE_ENSEMBLE_WEIGHTS,
        "pooled_oof_metrics": baseline_pooled_metrics,
        "mean_fold_roc_auc": float(baseline_fold_auc.mean()),
        "std_fold_roc_auc": float(baseline_fold_auc.std(ddof=1)),
        "pair_weighted_fold_roc_auc": pair_weighted_baseline_auc,
        "fold_auc": baseline_fold_auc.tolist(),
    },
    "six_model_equal_weight_ensemble_created": False,
    "advanced_blending_ready": True,
    "artifacts_directory": str(OUTPUT_DIR),
}
atomic_json(ENSEMBLE_DIR / "SIX_MODEL_EXPERIMENT_SUMMARY.json", summary)


# =============================================================================
# 17. AGGREGATE FEATURE IMPORTANCE
# =============================================================================

for model_name, importance_columns in [
    ("catboost_symmetric", ["IMPORTANCE"]),
    ("catboost_depthwise", ["IMPORTANCE"]),
    ("catboost_lossguide", ["IMPORTANCE"]),
    ("lightgbm", ["IMPORTANCE_GAIN", "IMPORTANCE_SPLIT"]),
    ("xgboost", ["IMPORTANCE_GAIN", "IMPORTANCE_WEIGHT"]),
]:
    importance_dir = IMPORTANCE_DIR / model_name.upper()
    frames = []
    for fold_number in range(1, N_SPLITS + 1):
        path = importance_dir / f"FOLD_{fold_number:02d}_IMPORTANCE.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["FOLD"] = fold_number
            frames.append(frame)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        atomic_csv(importance_dir / f"{model_name.upper()}_IMPORTANCE_BY_FOLD.csv", combined)
        mean_importance = combined.groupby("FEATURE", as_index=False)[importance_columns].mean()
        mean_importance = mean_importance.sort_values(importance_columns[0], ascending=False)
        atomic_csv(importance_dir / f"{model_name.upper()}_IMPORTANCE_MEAN_5FOLD.csv", mean_importance)


# =============================================================================
# 18. FINAL SUBMISSION VALIDATION AND MODEL MANIFEST
# =============================================================================

final_submission_path = ENSEMBLE_DIR / "SUBMISSION_FOUR_MODEL_BASELINE_EQUAL_WEIGHT.csv"
final_submission = pd.read_csv(final_submission_path)
if list(final_submission.columns) != [ID_COL, TARGET_COL] or final_submission.shape != (n_test, 2):
    raise RuntimeError("Four-model baseline submission has an invalid shape/schema")
if not np.array_equal(final_submission[ID_COL].to_numpy(), test_ids):
    raise RuntimeError("Baseline submission IDs/order do not match test.csv")
if final_submission[TARGET_COL].isna().any() or not final_submission[TARGET_COL].between(0, 1).all():
    raise RuntimeError("Baseline submission contains invalid probabilities")

manifest_entries = []
for fold_number in range(1, N_SPLITS + 1):
    # Reused imputation artifacts (not retrained in this run).
    for path in [IMPUTATION_VALUES_PATHS[fold_number], IMPUTATION_INDEX_PATHS[fold_number]]:
        manifest_entries.append(
            {
                "stage": "reused_imputation",
                "fold": fold_number,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    for source in EXT_SOURCE_COLUMNS:
        path = IMPUTER_MODEL_PATHS[(fold_number, source)]
        manifest_entries.append(
            {
                "stage": "reused_imputer_model",
                "fold": fold_number,
                "source": source,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    # GMM + preprocessing fold artifacts.
    for path in [
        GMM_DIR / f"FOLD_{fold_number:02d}" / "segmenter.pkl",
        GMM_DIR / f"FOLD_{fold_number:02d}" / "segmenter_metadata.json",
        GMM_DIR / f"FOLD_{fold_number:02d}" / "GMM_CACHE_STATUS.json",
        PREPROCESSING_DIR / f"FOLD_{fold_number:02d}" / "PREPROCESSING_BUNDLE.pkl",
        CACHE_DIR / f"FOLD_{fold_number:02d}" / "FEATURE_CACHE.pkl",
    ]:
        if not path.is_file():
            raise RuntimeError(f"Missing fold preprocessing/GMM artifact: {path}")
        manifest_entries.append(
            {
                "stage": "preprocessing_or_gmm",
                "fold": fold_number,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    # All six fold classification models and predictions.
    for model_name in MODEL_NAMES:
        for path in required_model_artifacts(model_name, fold_number):
            if not path.is_file():
                raise RuntimeError(f"Missing model artifact: {path}")
            manifest_entries.append(
                {
                    "stage": "classification",
                    "model": model_name,
                    "fold": fold_number,
                    "path": str(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )

# Register the CatBoost iteration-selection artifacts in the manifest.
if not catboost_selection_summary_path.is_file():
    raise RuntimeError("CatBoost inner-selection summary was not created")
manifest_entries.append(
    {
        "stage": "catboost_inner_selection_summary",
        "path": str(catboost_selection_summary_path),
        "size_bytes": int(catboost_selection_summary_path.stat().st_size),
        "sha256": sha256_file(catboost_selection_summary_path),
    }
)
for fold_number in range(1, N_SPLITS + 1):
    inner_split_path = CHECKPOINT_DIR / "CATBOOST_INNER_SPLITS" / f"FOLD_{fold_number:02d}.npz"
    if not inner_split_path.is_file():
        raise RuntimeError(f"Missing CatBoost inner split artifact: {inner_split_path}")
    manifest_entries.append(
        {
            "stage": "catboost_inner_selection_split",
            "fold": fold_number,
            "path": str(inner_split_path),
            "size_bytes": int(inner_split_path.stat().st_size),
            "sha256": sha256_file(inner_split_path),
        }
    )

manifest = {
    "status": "COMPLETE",
    "config_sha256": CONFIG_HASH,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "inputs": {"train": str(TRAIN_FILE), "test": str(TEST_FILE)},
    "imputation_models_reused": len(EXT_SOURCE_COLUMNS) * N_SPLITS,
    "gmm_segmenters": N_SPLITS,
    "classification_model_count": len(MODEL_NAMES) * N_SPLITS,
    "catboost_inner_selection_fit_count": 3 * N_SPLITS,
    "catboost_final_refit_count": 3 * N_SPLITS,
    "total_classifier_fit_calls_including_catboost_selection": len(MODEL_NAMES) * N_SPLITS + 3 * N_SPLITS,
    "classification_model_names": MODEL_NAMES,
    "all_artifacts": manifest_entries,
    "note": (
        "The four-model equal-weight ensemble is a baseline only. Six model OOF and fold-wise "
        "TEST predictions are retained for advanced blending; no six-model equal-weight blend is created."
    ),
}
atomic_json(OUTPUT_DIR / "MODEL_MANIFEST.json", manifest)

# Also publish the reference four-model submission at the experiment root.
root_submission = OUTPUT_DIR / "submission.csv"
atomic_csv(root_submission, final_submission)

mark_complete(
    CHECKPOINT_DIR / "EXPERIMENT_COMPLETE.json",
    "six_model_experiment_complete",
    CONFIG_HASH,
    [
        OUTPUT_DIR / "MODEL_MANIFEST.json",
        ENSEMBLE_DIR / "SIX_MODEL_EXPERIMENT_SUMMARY.json",
        ENSEMBLE_DIR / "OOF_PREDICTIONS_ALL_SIX_MODELS.csv",
        ENSEMBLE_DIR / "TEST_PREDICTIONS_ALL_SIX_MODELS_AND_FOLDS.csv",
        ENSEMBLE_DIR / "MODEL_FOLD_METRICS.csv",
        ENSEMBLE_DIR / "SIX_MODEL_FIVE_FOLD_METRICS_WITH_FOUR_MODEL_BASELINE.csv",
        catboost_selection_summary_path,
        *[
            CHECKPOINT_DIR / "CATBOOST_INNER_SPLITS" / f"FOLD_{fold_number:02d}.npz"
            for fold_number in range(1, N_SPLITS + 1)
        ],
        final_submission_path,
        root_submission,
    ],
)

print_section("SIX-MODEL TRAINING AND ARCHIVING COMPLETED")
print("Per-model pooled OOF ROC-AUC:")
for name in MODEL_NAMES:
    print(f"  {name:22s}: {model_oof_metrics[name]['roc_auc']:.9f}")
print(f"Four-model baseline pooled OOF ROC-AUC: {baseline_pooled_metrics['roc_auc']:.9f}")
print(f"Four-model baseline mean fold ROC-AUC : {baseline_fold_auc.mean():.9f}")
print(f"Four-model baseline std fold ROC-AUC  : {baseline_fold_auc.std(ddof=1):.9f}")
print(f"Four-model baseline pair-weighted AUC : {pair_weighted_baseline_auc:.9f}")
print(f"Reused EXT_SOURCE imputers            : {len(EXT_SOURCE_COLUMNS) * N_SPLITS}")
print(f"Saved classification models           : {len(MODEL_NAMES) * N_SPLITS}")
print(f"OOF predictions                       : {ENSEMBLE_DIR / 'OOF_PREDICTIONS_ALL_SIX_MODELS.csv'}")
print(f"TEST predictions                      : {ENSEMBLE_DIR / 'TEST_PREDICTIONS_ALL_SIX_MODELS_AND_FOLDS.csv'}")
print(f"Model manifest                        : {OUTPUT_DIR / 'MODEL_MANIFEST.json'}")
print(f"Baseline submission                   : {root_submission}")
print("No six-model equal-weight ensemble was created; predictions are ready for advanced blending.")
