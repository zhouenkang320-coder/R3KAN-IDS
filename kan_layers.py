import math
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_b_spline_basis(
    x: torch.Tensor, grid: torch.Tensor, order: int = 3
) -> torch.Tensor:
    batch_size, in_features = x.shape
    if grid.dim() == 1:
        x = torch.clamp(x, grid[0], grid[-1])
        grid_at = lambda idx: grid[idx]
        n_knots = len(grid)
    else:
        x = torch.max(torch.min(x, grid[:, -1].unsqueeze(0)), grid[:, 0].unsqueeze(0))
        grid_at = lambda idx: grid[:, idx].unsqueeze(0)
        n_knots = grid.shape[1]
    basis_list = []
    for k in range(n_knots - 1):
        b = ((x >= grid_at(k)) & (x < grid_at(k + 1))).float()
        if k == n_knots - 2:
            b = torch.logical_or(b, x == grid_at(n_knots - 1)).float()
        basis_list.append(b)
    for p in range(1, order + 1):
        new_basis = []
        for k in range(len(basis_list) - 1):
            left_denom = grid_at(k + p) - grid_at(k)
            left = torch.where(
                left_denom.abs() > 1e-10,
                (x - grid_at(k)) / left_denom.clamp_min(1e-10) * basis_list[k],
                torch.zeros_like(x),
            )
            right_denom = grid_at(k + p + 1) - grid_at(k + 1)
            right = torch.where(
                right_denom.abs() > 1e-10,
                (grid_at(k + p + 1) - x)
                / right_denom.clamp_min(1e-10)
                * basis_list[k + 1],
                torch.zeros_like(x),
            )
            new_basis.append(left + right)
        basis_list = new_basis
    return torch.stack(basis_list, dim=-1)


class KANLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_bases = grid_size + spline_order
        grid = torch.linspace(
            grid_range[0], grid_range[1], grid_size + 2 * spline_order + 1
        )
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, self.num_bases) * 0.1
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(
            self.spline_weight.view(self.out_features, -1), a=math.sqrt(5)
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_out = F.silu(x)
        base_out = F.linear(base_out, self.base_weight)
        basis = compute_b_spline_basis(x, self.grid, self.spline_order)
        spline_out = torch.einsum("oik,bik->bo", self.spline_weight, basis)
        return (base_out + spline_out, basis, spline_out)


class KAN(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_layers: int = 2,
        grid_size: int = 8,
        spline_order: int = 3,
        input_grid_range: Tuple[float, float] = (-1.0, 1.0),
        hidden_grid_range: Tuple[float, float] = (-3.0, 3.0),
        use_hidden_norm: bool = False,
    ):
        super().__init__()
        layers = []
        hidden_norms = []
        cur = in_features
        for i in range(num_layers):
            od = hidden_dim if i < num_layers - 1 else out_features
            hidden_norms.append(
                nn.LayerNorm(cur, elementwise_affine=False)
                if i > 0 and use_hidden_norm
                else nn.Identity()
            )
            grid_range = input_grid_range if i == 0 else hidden_grid_range
            layers.append(
                KANLinear(cur, od, grid_size, spline_order, grid_range=grid_range)
            )
            cur = od
        self.layers = nn.ModuleList(layers)
        self.hidden_norms = nn.ModuleList(hidden_norms)
        self.use_hidden_norm = use_hidden_norm

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        all_basis, all_spline = ([], [])
        for norm, layer in zip(self.hidden_norms, self.layers):
            x = norm(x)
            x, basis, spline = layer(x)
            all_basis.append(basis)
            all_spline.append(spline)
        return (x, all_basis, all_spline)
