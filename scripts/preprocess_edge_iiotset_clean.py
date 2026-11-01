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

FEATURE_COLUMNS = [
    "arp.opcode",
    "arp.hw.size",
    "icmp.checksum",
    "icmp.seq_le",
    "icmp.transmit_timestamp",
    "icmp.unused",
    "http.content_length",
    "http.response",
    "http.tls_port",
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.checksum",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.dstport",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
    "tcp.seq",
    "udp.port",
    "udp.stream",
    "udp.time_delta",
    "dns.qry.name",
    "dns.qry.qu",
    "dns.qry.type",
    "dns.retransmission",
    "dns.retransmit_request",
    "dns.retransmit_request_in",
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msg_decoded_as",
    "mqtt.msgtype",
    "mqtt.proto_len",
    "mqtt.topic_len",
    "mqtt.ver",
    "mbtcp.len",
    "mbtcp.trans_id",
    "mbtcp.unit_id",
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def split_indices(types: np.ndarray, test_size: float, val_size: float, seed: int):
    indices = np.arange(len(types))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=types
    )
    relative_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        random_state=seed,
        stratify=types[train_val_idx],
    )
    return (train_idx, val_idx, test_idx)


def distribution(values: np.ndarray) -> dict[str, int]:
    names, counts = np.unique(values.astype(str), return_counts=True)
    return {str(k): int(v) for k, v in zip(names, counts)}


def deduplicate_model_inputs(
    X: np.ndarray,
    y: np.ndarray,
    attack_type: np.ndarray,
    blocked_hashes: np.ndarray | None = None,
):
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
        "--input",
        default="data/Edge-IIoTset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv",
    )
    parser.add_argument(
        "--output",
        default="data/Edge-IIoTset/Selected dataset for ML and DL/DNN-EdgeIIoT-clean-v1.npz",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = output_path.with_suffix(".metadata.json")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.test_size <= 0 or args.val_size <= 0:
        raise ValueError("test-size and val-size must be positive")
    if args.test_size + args.val_size >= 1:
        raise ValueError("test-size + val-size must be less than 1")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    usecols = FEATURE_COLUMNS + ["Attack_label", "Attack_type"]
    dtype = {name: np.float32 for name in FEATURE_COLUMNS}
    dtype["Attack_label"] = np.int8
    print(f"Reading {input_path} ...", flush=True)
    df = pd.read_csv(input_path, usecols=usecols, dtype=dtype, low_memory=False)
    rows_loaded = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS + ["Attack_label", "Attack_type"]
    )
    rows_after_clean = len(df)
    feature_hash = pd.util.hash_pandas_object(
        df[FEATURE_COLUMNS], index=False
    ).to_numpy(dtype=np.uint64)
    labels = df["Attack_label"].to_numpy(dtype=np.int8)
    hash_label = pd.DataFrame({"hash": feature_hash, "label": labels})
    conflicting_hashes = (
        hash_label.groupby("hash", sort=False)["label"]
        .nunique()
        .loc[lambda x: x > 1]
        .index.to_numpy(dtype=np.uint64)
    )
    conflict_mask = np.isin(feature_hash, conflicting_hashes)
    conflicting_rows = int(conflict_mask.sum())
    if conflicting_rows:
        df = df.loc[~conflict_mask].copy()
        feature_hash = feature_hash[~conflict_mask]
    duplicate_mask = pd.Series(feature_hash).duplicated(keep="first").to_numpy()
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        df = df.loc[~duplicate_mask].copy()
        feature_hash = feature_hash[~duplicate_mask]
    df.reset_index(drop=True, inplace=True)
    constant_features = [
        name for name in FEATURE_COLUMNS if df[name].nunique(dropna=False) <= 1
    ]
    active_features = [
        name for name in FEATURE_COLUMNS if name not in constant_features
    ]
    X_raw = df[active_features].to_numpy(dtype=np.float32, copy=True)
    y = df["Attack_label"].to_numpy(dtype=np.int64, copy=True)
    attack_type = df["Attack_type"].astype(str).to_numpy()
    train_idx, val_idx, test_idx = split_indices(
        attack_type, args.test_size, args.val_size, args.seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx]).astype(np.float32)
    X_val = scaler.transform(X_raw[val_idx]).astype(np.float32)
    X_test = scaler.transform(X_raw[test_idx]).astype(np.float32)
    y_train, y_val, y_test = (y[train_idx], y[val_idx], y[test_idx])
    type_train = attack_type[train_idx].astype("U32")
    type_val = attack_type[val_idx].astype("U32")
    type_test = attack_type[test_idx].astype("U32")
    X_train, y_train, type_train, model_hash_train, train_internal_dup, _ = (
        deduplicate_model_inputs(X_train, y_train, type_train)
    )
    X_val, y_val, type_val, model_hash_val, val_internal_dup, val_blocked = (
        deduplicate_model_inputs(X_val, y_val, type_val, model_hash_train)
    )
    blocked_test = np.concatenate([model_hash_train, model_hash_val])
    X_test, y_test, type_test, model_hash_test, test_internal_dup, test_blocked = (
        deduplicate_model_inputs(X_test, y_test, type_test, blocked_test)
    )
    overlap = {
        "train_val": int(np.intersect1d(model_hash_train, model_hash_val).size),
        "train_test": int(np.intersect1d(model_hash_train, model_hash_test).size),
        "val_test": int(np.intersect1d(model_hash_val, model_hash_test).size),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Model-input overlap remains after splitting: {overlap}")
    print(f"Saving {output_path} ...", flush=True)
    np.savez_compressed(
        output_path,
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
        "version": "edge-iiot-clean-v1",
        "source": str(input_path.resolve()),
        "source_sha256": sha256_file(input_path),
        "output": str(output_path.resolve()),
        "source_numeric_feature_count": len(FEATURE_COLUMNS),
        "feature_count": len(active_features),
        "feature_names": active_features,
        "constant_features_removed": constant_features,
        "split": {
            "train": 1.0 - args.test_size - args.val_size,
            "validation": args.val_size,
            "test": args.test_size,
            "seed": args.seed,
            "stratify_by": "Attack_type",
        },
        "rows": {
            "loaded": rows_loaded,
            "after_nan_inf_removal": rows_after_clean,
            "conflicting_label_rows_removed": conflicting_rows,
            "exact_duplicate_rows_removed": duplicate_rows,
            "final_unique": len(df),
            "float32_internal_duplicates_removed": {
                "train": train_internal_dup,
                "validation": val_internal_dup,
                "test": test_internal_dup,
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
            "all": float(y.mean()),
            "train": float(y_train.mean()),
            "validation": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "attack_type_distribution": {
            "all": distribution(attack_type),
            "train": distribution(type_train),
            "validation": distribution(type_val),
            "test": distribution(type_test),
        },
        "exact_float32_model_input_overlap": overlap,
        "standardization": "StandardScaler fitted on train only",
        "elapsed_seconds": round(time.time() - started, 2),
        "limitations": [
            "Exact duplicate leakage is removed.",
            "Near-duplicate or capture-session correlation is not eliminated.",
            "Use a source/time-grouped split as an additional robustness test.",
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
