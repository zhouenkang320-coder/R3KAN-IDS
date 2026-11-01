from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class FeatureConstraints:
    minimum: np.ndarray
    maximum: np.ndarray
    binary_mask: np.ndarray


def infer_feature_constraints(
    X_train: np.ndarray, atol: float = 1e-07
) -> FeatureConstraints:
    values = np.asarray(X_train, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("X_train must be a non-empty two-dimensional array")
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    binary_mask = np.zeros(values.shape[1], dtype=bool)
    for column_index in range(values.shape[1]):
        low = minimum[column_index]
        high = maximum[column_index]
        if np.isclose(low, high, atol=atol, rtol=0.0):
            continue
        column = values[:, column_index]
        binary_mask[column_index] = bool(
            np.all(
                np.isclose(column, low, atol=atol, rtol=0.0)
                | np.isclose(column, high, atol=atol, rtol=0.0)
            )
        )
    return FeatureConstraints(minimum, maximum, binary_mask)


def project_feature_space(
    original: torch.Tensor,
    candidate: torch.Tensor,
    epsilon: float,
    *,
    feature_min: torch.Tensor | None = None,
    feature_max: torch.Tensor | None = None,
    binary_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(epsilon, torch.Tensor):
        radius = epsilon.float().to(original.device)
        if radius.dim() == 1:
            radius = radius.view(-1, 1)
    else:
        radius = max(0.0, float(epsilon))
    if (feature_min is None) != (feature_max is None):
        raise ValueError("feature_min and feature_max must be provided together")
    if feature_min is not None:
        minimum = feature_min.to(device=original.device, dtype=original.dtype)
        maximum = feature_max.to(device=original.device, dtype=original.dtype)
        lower = torch.maximum(original - radius, minimum)
        upper = torch.minimum(original + radius, maximum)
        feasible = lower <= upper
        projected = torch.maximum(torch.minimum(candidate, upper), lower)
        projected = torch.where(feasible, projected, original)
    else:
        projected = original + torch.clamp(candidate - original, -radius, radius)
    if binary_mask is not None:
        mask = binary_mask.to(device=projected.device, dtype=torch.bool)
        if mask.any():
            projected = torch.where(mask.unsqueeze(0), original, projected)
    return projected.detach()
