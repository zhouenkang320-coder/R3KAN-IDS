from pathlib import Path
import numpy as np

CACHES = {
    "CICIDS2017-Clean": "CICIDS2017/cicids2017_clean_v1.npz",
    "Edge-IIoTset-Clean": "Edge-IIoTset/Selected dataset for ML and DL/DNN-EdgeIIoT-clean-v1.npz",
    "CIC-IoT-DIAD-2024-Clean": "CIC-IoT-DIAD-2024/cic_iot_diad_2024_clean_v1.npz",
}


def load_splits(dataset, data_dir):
    path = Path(data_dir) / CACHES[dataset]
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset cache not found: {path}. Run the corresponding preprocessing script first."
        )
    with np.load(path, allow_pickle=False) as cache:
        result = {}
        for split in ["train", "val", "test"]:
            x = cache[f"X_{split}"].astype(np.float32, copy=False)
            y = cache[f"y_{split}"].astype(np.int64, copy=False)
            if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or (len(y) == 0):
                raise ValueError(f"Invalid {split} dimensions")
            if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
                raise ValueError(f"Invalid {split} values")
            if len(np.unique(y)) != 2:
                raise ValueError(f"{split} must contain both Normal and Attack samples")
            result[f"X_{split}"], result[f"y_{split}"] = (x, y)
        if len({result[f"X_{s}"].shape[1] for s in ["train", "val", "test"]}) != 1:
            raise ValueError("Feature dimensions differ across splits")
        key = "attack_type_train"
        values = cache[key] if key in cache.files else result["y_train"]
        if "attack_type_names" in cache.files and np.issubdtype(
            values.dtype, np.integer
        ):
            values = cache["attack_type_names"].astype(str)[values.astype(np.int64)]
        types = []
        for item in values.astype(str):
            text = item.replace("\ufffd", "-").replace("锟?", "-")
            if text.startswith("Web Attack"):
                if "Brute Force" in text:
                    text = "Web Attack - Brute Force"
                elif "Sql Injection" in text or "SQL Injection" in text:
                    text = "Web Attack - SQL Injection"
                elif "XSS" in text:
                    text = "Web Attack - XSS"
            types.append(" ".join(text.split()))
        result[key] = np.asarray(types, dtype="U128")
        if len(result[key]) != len(result["y_train"]):
            raise ValueError("Attack subtype labels do not match training samples")
    return result


def attack_subset_indices(labels, attack_types, ratio, seed):
    if not 0 < ratio < 1:
        raise ValueError("attack_ratio must be in (0, 1)")
    normal = np.flatnonzero(labels == 0)
    attack = np.flatnonzero(labels == 1)
    if len(normal) == 0 or len(attack) == 0:
        raise ValueError("Training data must contain both classes")
    rng = np.random.default_rng(seed)
    groups, priorities = ([], [])
    for subtype in np.unique(attack_types[attack]):
        group = attack[attack_types[attack] == subtype]
        group = rng.permutation(group)
        priority = (
            np.arange(len(group), dtype=np.float64) + rng.random(len(group))
        ) / len(group)
        groups.append(group)
        priorities.append(priority)
    all_indices = np.concatenate(groups)
    order = all_indices[np.argsort(np.concatenate(priorities), kind="stable")]
    needed = int(round(ratio / (1 - ratio) * len(normal)))
    if needed < 1 or needed > len(attack):
        raise ValueError(
            "Requested Attack proportion cannot be obtained by downsampling attacks only"
        )
    return np.concatenate([normal, order[:needed]])
