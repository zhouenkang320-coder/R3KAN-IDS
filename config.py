from dataclasses import asdict, dataclass, fields
from pathlib import Path
import yaml


@dataclass
class Config:
    dataset: str = "CICIDS2017-Clean"
    data_dir: str = "./data"
    output_dir: str = "./runs"
    device: str = "cuda"
    random_seed: int = 42
    total_timesteps: int = 1000000
    learning_rate: float = 0.0005
    batch_size: int = 512
    minibatch_size: int = 256
    update_epochs: int = 6
    max_grad_norm: float = 0.5
    advantage_normalization: bool = True
    c_1: float = 0.5
    c_2: float = 0.15
    epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    target_kl: float = 0.03
    kan_hidden_dim: int = 64
    kan_num_layers: int = 2
    kan_grid_size: int = 12
    kan_spline_order: int = 3
    kan_input_grid_range: float = 1.0
    kan_hidden_grid_range: float = 3.0
    kan_use_hidden_norm: bool = True
    critic_hidden_dim: int = 128
    use_rca: bool = True
    rc_target_fpr: float = 0.01
    rc_target_fnr: float = 0.05
    rc_dual_lr: float = 0.05
    rc_lambda_max: float = 3.0
    rc_ema_alpha: float = 0.05
    rc_warmup_steps: int = 30000
    rc_cost_adv_clip: float = 5.0
    use_rapc: bool = True
    rapc_epsilon: float = 0.02
    rapc_inner_steps: int = 3
    rapc_fraction: float = 0.25
    rapc_coef: float = 0.02
    rapc_warmup_steps: int = 50000
    rapc_local_risk_mode: str = "max"
    selection_metric: str = "clean_f1"
    eval_interval: int = 25000
    eval_batch_size: int = 512
    max_validation_samples: int = 50000
    evaluation_seed: int = 3407

    def validate(self):
        if self.dataset not in {
            "CICIDS2017-Clean",
            "Edge-IIoTset-Clean",
            "CIC-IoT-DIAD-2024-Clean",
        }:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.rapc_local_risk_mode not in {"max", "p", "v", "random"}:
            raise ValueError("rapc_local_risk_mode must be max, p, v, or random")
        if self.selection_metric not in {"clean_f1", "pgd10_f1"}:
            raise ValueError("selection_metric must be clean_f1 or pgd10_f1")
        if not isinstance(self.batch_size, int) or self.batch_size < 2:
            raise ValueError("batch_size must be an integer >= 2")
        if not isinstance(self.minibatch_size, int) or self.minibatch_size < 2:
            raise ValueError("minibatch_size must be an integer >= 2")
        if self.batch_size % self.minibatch_size:
            raise ValueError("batch_size must be divisible by minibatch_size")
        if (
            min(
                self.total_timesteps,
                self.update_epochs,
                self.eval_interval,
                self.eval_batch_size,
            )
            <= 0
        ):
            raise ValueError("Training and evaluation budgets must be positive")
        if (
            min(
                self.learning_rate,
                self.epsilon,
                self.max_grad_norm,
                self.value_clip_epsilon,
            )
            <= 0
        ):
            raise ValueError("Learning rate and optimization scales must be positive")
        if not 0 <= self.rapc_fraction <= 1:
            raise ValueError("rapc_fraction must be in [0, 1]")
        if self.rapc_epsilon <= 0 or self.rapc_inner_steps < 1 or self.rapc_coef < 0:
            raise ValueError("Invalid RAPC radius, iteration count, or coefficient")
        if self.rapc_warmup_steps < 0 or self.rc_warmup_steps < 0:
            raise ValueError("Warm-up counts must be nonnegative")
        if (
            not 0 <= self.rc_ema_alpha <= 1
            or self.rc_dual_lr < 0
            or self.rc_lambda_max < 0
        ):
            raise ValueError("Invalid RCA controller parameters")

    def to_dict(self):
        return asdict(self)


def make_config(values):
    allowed = {f.name for f in fields(Config)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = Config(**values)
    config.validate()
    return config


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return make_config(yaml.safe_load(handle) or {})
