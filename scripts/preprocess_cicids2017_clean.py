from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(codes: np.ndarray, names: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(codes, return_counts=True)
    return {str(names[int(value)]): int(count) for value, count in zip(values, counts)}


def robust_stratify_codes(codes: np.ndarray, minimum_count: int = 5) -> np.ndarray:
    values, counts = np.unique(codes, return_counts=True)
    count_map = dict(zip(values.tolist(), counts.tolist()))
    fallback = np.where(codes == 0, -2, -1)
    return np.asarray(
        [
            code if count_map[int(code)] >= minimum_count else fallback[i]
            for i, code in enumerate(codes)
        ],
        dtype=np.int32,
    )


def split_indices(types: np.ndarray, test_size: float, val_size: float, seed: int):
    indices = np.arange(len(types))
    strata = robust_stratify_codes(types)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=strata
    )
    relative_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        random_state=seed,
        stratify=strata[train_val_idx],
    )
    return (train_idx, val_idx, test_idx)


def model_input_hash(X: np.ndarray) -> np.ndarray:
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy(
        dtype=np.uint64
    )


def deduplicate_model_inputs(
    X: np.ndarray,
    y: np.ndarray,
    attack_type: np.ndarray,
    blocked_hashes: np.ndarray | None = None,
):
    hashes = model_input_hash(X)
    internal_duplicate = pd.Series(hashes).duplicated(keep="first").to_numpy()
    blocked = (
        np.isin(hashes, blocked_hashes)
        if blocked_hashes is not None and len(blocked_hashes)
        else np.zeros(len(hashes), dtype=bool)
    )
    keep = ~(internal_duplicate | blocked)
    return (
        X[keep],
        y[keep],
        attack_type[keep],
        hashes[keep],
        int(internal_duplicate.sum()),
        int((blocked & ~internal_duplicate).sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/CICIDS2017/MachineLearningCVE")
    parser.add_argument("--output", default="data/CICIDS2017/cicids2017_clean_v1.npz")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    metadata_path = output_path.with_suffix(".metadata.json")
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CICIDS2017 CSV files found in {input_dir}")
    if args.test_size <= 0 or args.val_size <= 0:
        raise ValueError("test-size and val-size must be positive")
    if args.test_size + args.val_size >= 1:
        raise ValueError("test-size + val-size must be less than 1")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frames: list[pd.DataFrame] = []
    rows_loaded = 0
    feature_names: list[str] | None = None
    for path in csv_paths:
        print(f"Reading {path.name} ...", flush=True)
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        if "Label" not in frame.columns:
            raise ValueError(f"Label column not found in {path}")
        current_features = [column for column in frame.columns if column != "Label"]
        if feature_names is None:
            feature_names = current_features
        elif current_features != feature_names:
            raise ValueError(f"Feature columns differ in {path.name}")
        rows_loaded += len(frame)
        numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.astype(np.float32)
        labels = frame["Label"].astype(str).str.strip()
        cleaned = numeric.copy()
        cleaned["__label__"] = labels
        cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
        cleaned.dropna(subset=feature_names + ["__label__"], inplace=True)
        frames.append(cleaned)
        del frame, numeric, cleaned
    assert feature_names is not None
    df = pd.concat(frames, ignore_index=True)
    del frames
    rows_after_clean = len(df)
    label_text = df.pop("__label__").astype(str).str.strip()
    attack_names, attack_codes = np.unique(label_text.to_numpy(), return_inverse=True)
    attack_names = attack_names.astype("U96")
    attack_codes = attack_codes.astype(np.int16)
    benign_codes = np.flatnonzero(np.char.upper(attack_names) == "BENIGN")
    if len(benign_codes) != 1:
        raise ValueError(
            f"Expected exactly one BENIGN label, got {attack_names.tolist()}"
        )
    benign_code = int(benign_codes[0])
    y = (attack_codes != benign_code).astype(np.int8)
    feature_hash = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)
    hash_label = pd.DataFrame({"hash": feature_hash, "label": y})
    conflicting_hashes = (
        hash_label.groupby("hash", sort=False)["label"]
        .nunique()
        .loc[lambda values: values > 1]
        .index.to_numpy(dtype=np.uint64)
    )
    conflict_mask = np.isin(feature_hash, conflicting_hashes)
    conflicting_rows = int(conflict_mask.sum())
    if conflicting_rows:
        keep = ~conflict_mask
        df = df.loc[keep].copy()
        y = y[keep]
        attack_codes = attack_codes[keep]
        feature_hash = feature_hash[keep]
    duplicate_mask = pd.Series(feature_hash).duplicated(keep="first").to_numpy()
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        keep = ~duplicate_mask
        df = df.loc[keep].copy()
        y = y[keep]
        attack_codes = attack_codes[keep]
    df.reset_index(drop=True, inplace=True)
    constant_features = [
        name for name in feature_names if df[name].nunique(dropna=False) <= 1
    ]
    active_features = [name for name in feature_names if name not in constant_features]
    X_raw = df[active_features].to_numpy(dtype=np.float32, copy=True)
    del df
    ordered_names = np.concatenate(
        [
            attack_names[[benign_code]],
            attack_names[np.arange(len(attack_names)) != benign_code],
        ]
    )
    old_to_new = np.empty(len(attack_names), dtype=np.int16)
    for new_code, name in enumerate(ordered_names):
        old_to_new[int(np.flatnonzero(attack_names == name)[0])] = new_code
    attack_codes = old_to_new[attack_codes]
    attack_names = ordered_names
    train_idx, val_idx, test_idx = split_indices(
        attack_codes, args.test_size, args.val_size, args.seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx]).astype(np.float32)
    X_val = scaler.transform(X_raw[val_idx]).astype(np.float32)
    X_test = scaler.transform(X_raw[test_idx]).astype(np.float32)
    y_train, y_val, y_test = (y[train_idx], y[val_idx], y[test_idx])
    type_train = attack_codes[train_idx]
    type_val = attack_codes[val_idx]
    type_test = attack_codes[test_idx]
    del X_raw
    X_train, y_train, type_train, hash_train, tr_dup, _ = deduplicate_model_inputs(
        X_train, y_train, type_train
    )
    X_val, y_val, type_val, hash_val, val_dup, val_blocked = deduplicate_model_inputs(
        X_val, y_val, type_val, hash_train
    )
    X_test, y_test, type_test, hash_test, test_dup, test_blocked = (
        deduplicate_model_inputs(
            X_test, y_test, type_test, np.concatenate([hash_train, hash_val])
        )
    )
    overlap = {
        "train_val": int(np.intersect1d(hash_train, hash_val).size),
        "train_test": int(np.intersect1d(hash_train, hash_test).size),
        "val_test": int(np.intersect1d(hash_val, hash_test).size),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Model-input overlap remains: {overlap}")
    print(f"Saving {output_path} ...", flush=True)
    np.savez_compressed(
        output_path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        attack_type_train=type_train.astype(np.int16),
        attack_type_val=type_val.astype(np.int16),
        attack_type_test=type_test.astype(np.int16),
        attack_type_names=attack_names,
        feature_names=np.asarray(active_features, dtype="U64"),
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        split_seed=np.asarray(args.seed, dtype=np.int64),
    )
    metadata = {
        "version": "cicids2017-clean-v1",
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in csv_paths
        ],
        "output": str(output_path.resolve()),
        "feature_count": len(active_features),
        "feature_names": active_features,
        "constant_features_removed": constant_features,
        "split": {
            "train": 1.0 - args.test_size - args.val_size,
            "validation": args.val_size,
            "test": args.test_size,
            "seed": args.seed,
            "stratify_by": "attack subtype (rare subtypes use binary fallback)",
        },
        "rows": {
            "loaded": rows_loaded,
            "after_nan_inf_removal": rows_after_clean,
            "conflicting_binary_label_rows_removed": conflicting_rows,
            "exact_duplicate_rows_removed": duplicate_rows,
            "float32_internal_duplicates_removed": {
                "train": tr_dup,
                "validation": val_dup,
                "test": test_dup,
            },
            "float32_cross_split_rows_removed": {
                "validation": val_blocked,
                "test": test_blocked,
            },
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "binary_attack_ratio": {
            "train": float(y_train.mean()),
            "validation": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "attack_type_distribution": {
            "train": distribution(type_train, attack_names),
            "validation": distribution(type_val, attack_names),
            "test": distribution(type_test, attack_names),
        },
        "exact_float32_model_input_overlap": overlap,
        "standardization": "StandardScaler fitted on train only",
        "elapsed_seconds": round(time.time() - started, 2),
        "limitations": [
            "Exact duplicate leakage is removed.",
            "Near-duplicate and capture-day correlation are not eliminated.",
            "A day- or session-grouped split should be added as a robustness experiment.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata["rows"], indent=2))
    print("attack ratios:", metadata["binary_attack_ratio"])
    print("overlap:", overlap)
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
