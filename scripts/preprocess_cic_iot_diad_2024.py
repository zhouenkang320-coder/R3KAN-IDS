from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DROP_COLUMNS = {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"}


def normalize_attack_type(path: Path, class_name: str) -> str:
    if class_name.lower() == "benign":
        return "Benign"
    name = re.sub("\\.pcap_Flow\\.csv$", "", path.name, flags=re.IGNORECASE)
    return re.sub("\\d+$", "", name)


def discover_csvs(root: Path) -> list[tuple[Path, str, int, str]]:
    items: list[tuple[Path, str, int, str]] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        class_name = class_dir.name
        label = 0 if class_name.lower() == "benign" else 1
        for csv_path in sorted(class_dir.glob("*.csv")):
            items.append(
                (
                    csv_path,
                    class_name,
                    label,
                    normalize_attack_type(csv_path, class_name),
                )
            )
    if not items:
        raise FileNotFoundError(f"No CSV files found under {root}")
    return items


def get_feature_columns(first_csv: Path) -> list[str]:
    header = pd.read_csv(first_csv, nrows=0)
    return [c for c in header.columns if c not in DROP_COLUMNS]


def clean_numeric_chunk(
    chunk: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    chunk = chunk.reindex(columns=feature_columns)
    for col in feature_columns:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
    return chunk.astype(np.float32, copy=False)


def distribution(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values.astype(str), return_counts=True)
    return {str(k): int(v) for k, v in zip(keys, counts)}


def count_valid_rows(
    items: list[tuple[Path, str, int, str]], feature_columns: list[str], chunksize: int
) -> tuple[dict[str, int], dict[str, int], dict[str, dict]]:
    count_by_type: dict[str, int] = {}
    count_by_binary = {"0": 0, "1": 0}
    file_stats: dict[str, dict] = {}
    usecols = feature_columns + ["Label"]
    print("\nPass 1/2: counting valid rows ...", flush=True)
    for file_idx, (csv_path, class_name, label, attack_type) in enumerate(
        items, start=1
    ):
        loaded = 0
        kept = 0
        print(
            f"[count {file_idx:03d}/{len(items):03d}] {class_name}/{csv_path.name}",
            flush=True,
        )
        for chunk in pd.read_csv(
            csv_path,
            usecols=lambda c: c in usecols,
            chunksize=chunksize,
            low_memory=False,
        ):
            loaded += len(chunk)
            kept += len(clean_numeric_chunk(chunk, feature_columns))
        count_by_type[attack_type] = count_by_type.get(attack_type, 0) + kept
        count_by_binary[str(label)] += kept
        file_stats[str(csv_path)] = {
            "class": class_name,
            "binary_label": int(label),
            "attack_type": attack_type,
            "rows_loaded": int(loaded),
            "rows_after_numeric_clean": int(kept),
            "rows_removed_nan_inf": int(loaded - kept),
            "file_size": int(csv_path.stat().st_size),
        }
    return (count_by_type, count_by_binary, file_stats)


def build_sampling_plan(
    count_by_type: dict[str, int],
    target_attack_ratio: float | None,
    max_rows: int | None,
) -> dict[str, int]:
    total_available = sum(count_by_type.values())
    benign_count = count_by_type.get("Benign", 0)
    attack_count = total_available - benign_count
    if benign_count <= 0 or attack_count <= 0:
        raise ValueError("Both benign and attack rows are required.")
    if target_attack_ratio is None and max_rows is None:
        raise ValueError(
            "Set --target-attack-ratio and/or --max-rows for CIC IoT-DIAD 2024."
        )
    if target_attack_ratio is not None:
        if not 0.0 < target_attack_ratio < 1.0:
            raise ValueError("--target-attack-ratio must be in (0, 1)")
        max_total_by_benign = int(np.floor(benign_count / (1.0 - target_attack_ratio)))
        max_total_by_attack = int(np.floor(attack_count / target_attack_ratio))
        total = min(max_total_by_benign, max_total_by_attack)
        if max_rows is not None:
            total = min(total, max_rows)
        attack_budget = int(round(total * target_attack_ratio))
        benign_budget = total - attack_budget
    else:
        total = min(total_available, int(max_rows))
        benign_budget = int(round(total * benign_count / total_available))
        attack_budget = total - benign_budget
    plan = {"Benign": min(benign_budget, benign_count)}
    attack_types = [t for t in sorted(count_by_type) if t != "Benign"]
    attack_total_available = sum((count_by_type[t] for t in attack_types))
    remaining = min(attack_budget, attack_total_available)
    raw_alloc = {
        t: attack_budget * count_by_type[t] / attack_total_available
        for t in attack_types
    }
    for t in attack_types:
        plan[t] = min(count_by_type[t], int(np.floor(raw_alloc[t])))
    assigned = sum((plan[t] for t in attack_types))
    leftovers = sorted(
        attack_types, key=lambda t: raw_alloc[t] - np.floor(raw_alloc[t]), reverse=True
    )
    for t in leftovers:
        if assigned >= remaining:
            break
        if plan[t] < count_by_type[t]:
            plan[t] += 1
            assigned += 1
    return plan


def read_sampled_rows(
    items: list[tuple[Path, str, int, str]],
    feature_columns: list[str],
    chunksize: int,
    plan: dict[str, int],
    count_by_type: dict[str, int],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    remaining_need = {k: int(v) for k, v in plan.items()}
    remaining_available = {k: int(v) for k, v in count_by_type.items()}
    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    attack_types: list[np.ndarray] = []
    sampled_by_type = {k: 0 for k in plan}
    usecols = feature_columns + ["Label"]
    print("\nPass 2/2: streaming exact sampling ...", flush=True)
    for file_idx, (csv_path, class_name, label, attack_type) in enumerate(
        items, start=1
    ):
        print(
            f"[sample {file_idx:03d}/{len(items):03d}] {class_name}/{csv_path.name}",
            flush=True,
        )
        for chunk in pd.read_csv(
            csv_path,
            usecols=lambda c: c in usecols,
            chunksize=chunksize,
            low_memory=False,
        ):
            chunk = clean_numeric_chunk(chunk, feature_columns)
            m = len(chunk)
            if m == 0:
                continue
            need = remaining_need.get(attack_type, 0)
            avail = remaining_available.get(attack_type, 0)
            if need <= 0 or avail <= 0:
                remaining_available[attack_type] = max(0, avail - m)
                continue
            k = int(rng.hypergeometric(need, max(0, avail - need), m))
            k = min(k, m, need)
            if k > 0:
                chosen_pos = rng.choice(m, size=k, replace=False)
                chosen = chunk.iloc[chosen_pos].reset_index(drop=True)
                frames.append(chosen)
                labels.append(np.full(k, label, dtype=np.int8))
                attack_types.append(np.full(k, attack_type, dtype=object))
                sampled_by_type[attack_type] = sampled_by_type.get(attack_type, 0) + k
                remaining_need[attack_type] -= k
            remaining_available[attack_type] = max(0, avail - m)
    if not frames:
        raise RuntimeError("No rows were sampled.")
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(labels).astype(np.int8),
        np.concatenate(attack_types).astype(object),
        {
            "sampling_plan": {k: int(v) for k, v in plan.items()},
            "sampled_by_type": {k: int(v) for k, v in sampled_by_type.items()},
            "sampling_shortfall": {k: int(remaining_need.get(k, 0)) for k in plan},
        },
    )


def remove_conflicts_and_duplicates(
    X_df: pd.DataFrame, y: np.ndarray, attack_type: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    hashes = pd.util.hash_pandas_object(X_df, index=False).to_numpy(dtype=np.uint64)
    hash_label = pd.DataFrame({"hash": hashes, "label": y})
    conflicting_hashes = (
        hash_label.groupby("hash", sort=False)["label"]
        .nunique()
        .loc[lambda s: s > 1]
        .index.to_numpy(dtype=np.uint64)
    )
    conflict_mask = np.isin(hashes, conflicting_hashes)
    duplicate_mask = pd.Series(hashes).duplicated(keep="first").to_numpy()
    keep = ~(conflict_mask | duplicate_mask)
    return (
        X_df.loc[keep].reset_index(drop=True),
        y[keep],
        attack_type[keep],
        {
            "conflicting_feature_rows_removed": int(conflict_mask.sum()),
            "duplicate_feature_rows_removed": int(
                (duplicate_mask & ~conflict_mask).sum()
            ),
            "rows_after_dedup": int(keep.sum()),
        },
    )


def split_indices(
    attack_type: np.ndarray, test_size: float, val_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(attack_type))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=attack_type
    )
    relative_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        random_state=seed,
        stratify=attack_type[train_val_idx],
    )
    return (train_idx, val_idx, test_idx)


def deduplicate_model_inputs(
    X: np.ndarray,
    y: np.ndarray,
    attack_type: np.ndarray,
    blocked_hashes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    hashes = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy(
        dtype=np.uint64
    )
    duplicate_mask = pd.Series(hashes).duplicated(keep="first").to_numpy()
    blocked_mask = (
        np.isin(hashes, blocked_hashes)
        if blocked_hashes is not None and len(blocked_hashes)
        else np.zeros(len(hashes), dtype=bool)
    )
    keep = ~(duplicate_mask | blocked_mask)
    return (
        X[keep],
        y[keep],
        attack_type[keep],
        hashes[keep],
        int(duplicate_mask.sum()),
        int((blocked_mask & ~duplicate_mask).sum()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=Path("data/CIC-IoT-DIAD-2024")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/CIC-IoT-DIAD-2024/cic_iot_diad_2024_clean_v1.npz"),
    )
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=1500000)
    parser.add_argument("--target-attack-ratio", type=float, default=0.3)
    args = parser.parse_args()
    started = time.time()
    if not args.input_root.exists():
        raise FileNotFoundError(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(".metadata.json")
    items = discover_csvs(args.input_root)
    feature_columns = get_feature_columns(items[0][0])
    print(f"Found {len(items)} CSV files.")
    print(f"Initial feature columns: {len(feature_columns)}")
    print(f"Target attack ratio: {args.target_attack_ratio}; max rows: {args.max_rows}")
    count_by_type, count_by_binary, file_stats = count_valid_rows(
        items, feature_columns, args.chunksize
    )
    plan = build_sampling_plan(count_by_type, args.target_attack_ratio, args.max_rows)
    print("\nValid rows by type:")
    for key, value in sorted(count_by_type.items()):
        print(f"  {key}: {value:,d}")
    print("Sampling plan:")
    for key, value in sorted(plan.items()):
        print(f"  {key}: {value:,d}")
    X_df, y, attack_type, sampling_stats = read_sampled_rows(
        items, feature_columns, args.chunksize, plan, count_by_type, args.seed
    )
    X_df, y, attack_type, dedup_stats = remove_conflicts_and_duplicates(
        X_df, y, attack_type
    )
    constant_features = [c for c in X_df.columns if X_df[c].nunique(dropna=False) <= 1]
    if constant_features:
        X_df = X_df.drop(columns=constant_features)
    active_features = list(X_df.columns)
    print("\nAfter clean/dedup:")
    print(
        f"  rows={len(y):,d}, features={len(active_features)}, attack_ratio={y.mean():.4f}"
    )
    for key, value in distribution(attack_type).items():
        print(f"  {key}: {value:,d}")
    train_idx, val_idx, test_idx = split_indices(
        attack_type, args.test_size, args.val_size, args.seed
    )
    X_raw = X_df.to_numpy(dtype=np.float32, copy=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx]).astype(np.float32)
    X_val = scaler.transform(X_raw[val_idx]).astype(np.float32)
    X_test = scaler.transform(X_raw[test_idx]).astype(np.float32)
    y_train = y[train_idx].astype(np.int64)
    y_val = y[val_idx].astype(np.int64)
    y_test = y[test_idx].astype(np.int64)
    type_train = attack_type[train_idx].astype("U64")
    type_val = attack_type[val_idx].astype("U64")
    type_test = attack_type[test_idx].astype("U64")
    X_train, y_train, type_train, hash_train, tr_dup, _ = deduplicate_model_inputs(
        X_train, y_train, type_train
    )
    X_val, y_val, type_val, hash_val, val_dup, val_overlap = deduplicate_model_inputs(
        X_val, y_val, type_val, hash_train
    )
    blocked_test = np.concatenate([hash_train, hash_val])
    X_test, y_test, type_test, hash_test, te_dup, te_overlap = deduplicate_model_inputs(
        X_test, y_test, type_test, blocked_test
    )
    overlap = {
        "train_val": int(np.intersect1d(hash_train, hash_val).size),
        "train_test": int(np.intersect1d(hash_train, hash_test).size),
        "val_test": int(np.intersect1d(hash_val, hash_test).size),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Model-input overlap remains: {overlap}")
    print(f"\nSaving {args.output} ...", flush=True)
    np.savez_compressed(
        args.output,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        attack_type_train=type_train,
        attack_type_val=type_val,
        attack_type_test=type_test,
        feature_names=np.asarray(active_features, dtype="U64"),
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        split_seed=np.asarray(args.seed, dtype=np.int64),
    )
    metadata = {
        "version": "cic-iot-diad-2024-clean-v1",
        "source_root": str(args.input_root),
        "output": str(args.output),
        "feature_count": len(active_features),
        "feature_names": active_features,
        "constant_features_removed": constant_features,
        "binary_label_rule": "Benign folder -> 0; all other folders -> 1",
        "valid_rows_by_binary_before_sampling": count_by_binary,
        "valid_rows_by_type_before_sampling": {
            k: int(v) for k, v in count_by_type.items()
        },
        "file_stats": file_stats,
        "sampling": {
            "target_attack_ratio": args.target_attack_ratio,
            "max_rows": args.max_rows,
            **sampling_stats,
        },
        "dedup": dedup_stats,
        "split": {
            "train": 1.0 - args.test_size - args.val_size,
            "validation": args.val_size,
            "test": args.test_size,
            "seed": args.seed,
            "stratify_by": "attack_type",
        },
        "rows": {
            "all_after_dedup": int(len(y)),
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
            "model_input_internal_duplicates_removed": {
                "train": tr_dup,
                "validation": val_dup,
                "test": te_dup,
            },
            "model_input_cross_split_duplicates_removed": {
                "validation_vs_train": val_overlap,
                "test_vs_train_val": te_overlap,
            },
        },
        "distribution": {
            "all_binary": distribution(y),
            "all_attack_type": distribution(attack_type),
            "train_binary": distribution(y_train),
            "validation_binary": distribution(y_val),
            "test_binary": distribution(y_test),
            "train_attack_type": distribution(type_train),
            "validation_attack_type": distribution(type_val),
            "test_attack_type": distribution(type_test),
        },
        "standardization": "StandardScaler fitted on train only",
        "overlap_check": overlap,
        "elapsed_sec": round(time.time() - started, 2),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("\nDone.")
    print(f"  output: {args.output}")
    print(f"  metadata: {metadata_path}")
    print(f"  train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(
        f"  attack ratio: train={y_train.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}"
    )
    print(f"  overlap: {overlap}")


if __name__ == "__main__":
    main()
