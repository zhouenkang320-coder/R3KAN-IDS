from typing import Optional
import torch


class RiskConstrainedAdvantage:

    def __init__(
        self,
        target_fpr: float = 0.01,
        target_fnr: float = 0.05,
        dual_lr: float = 0.05,
        lambda_fp_init: float = 0.0,
        lambda_fn_init: float = 0.0,
        lambda_max: float = 3.0,
        ema_alpha: float = 0.05,
        warmup_steps: int = 30000,
        cost_adv_clip: float = 5.0,
    ):
        self.target_fpr = max(float(target_fpr), 0.0)
        self.target_fnr = max(float(target_fnr), 0.0)
        self.dual_lr = max(float(dual_lr), 0.0)
        self.lambda_max = max(float(lambda_max), 0.0)
        self.lambda_fp = min(max(float(lambda_fp_init), 0.0), self.lambda_max)
        self.lambda_fn = min(max(float(lambda_fn_init), 0.0), self.lambda_max)
        self.ema_alpha = min(max(float(ema_alpha), 0.0), 1.0)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.cost_adv_clip = max(float(cost_adv_clip), 0.0)
        self.fpr_ema: Optional[float] = None
        self.fnr_ema: Optional[float] = None
        self.last_fpr = 0.0
        self.last_fnr = 0.0
        self.fp_violation = 0.0
        self.fn_violation = 0.0
        self.last_fp_adv_mean = 0.0
        self.last_fn_adv_mean = 0.0
        self.last_fp_adv_std = 0.0
        self.last_fn_adv_std = 0.0
        self.last_correction_mean = 0.0
        self.last_correction_std = 0.0

    def _ema(self, previous: Optional[float], value: float) -> float:
        if previous is None:
            return float(value)
        return (1.0 - self.ema_alpha) * previous + self.ema_alpha * float(value)

    def _warmup_strength(self, global_step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, float(global_step) / self.warmup_steps))

    def compute(
        self,
        task_advantages: torch.Tensor,
        actions: torch.Tensor,
        labels: torch.Tensor,
        feedback_actions: torch.Tensor,
        global_step: int,
    ) -> torch.Tensor:
        labels = labels.long()
        actions = actions.long()
        feedback_actions = feedback_actions.long()
        normal_mask = labels == 0
        attack_mask = labels == 1
        with torch.no_grad():
            batch_fpr = (
                float((feedback_actions[normal_mask] == 1).float().mean().item())
                if normal_mask.any()
                else self.last_fpr
            )
            batch_fnr = (
                float((feedback_actions[attack_mask] == 0).float().mean().item())
                if attack_mask.any()
                else self.last_fnr
            )
            self.last_fpr = batch_fpr
            self.last_fnr = batch_fnr
            self.fpr_ema = self._ema(self.fpr_ema, batch_fpr)
            self.fnr_ema = self._ema(self.fnr_ema, batch_fnr)
            self.fp_violation = max(0.0, float(self.fpr_ema) - self.target_fpr)
            self.fn_violation = max(0.0, float(self.fnr_ema) - self.target_fnr)
            strength = self._warmup_strength(global_step)
            if strength >= 1.0 and self.dual_lr > 0.0:
                self.lambda_fp = min(
                    self.lambda_max,
                    max(
                        0.0,
                        self.lambda_fp
                        + self.dual_lr * (float(self.fpr_ema) - self.target_fpr),
                    ),
                )
                self.lambda_fn = min(
                    self.lambda_max,
                    max(
                        0.0,
                        self.lambda_fn
                        + self.dual_lr * (float(self.fnr_ema) - self.target_fnr),
                    ),
                )
        normal_fraction = normal_mask.float().mean().clamp_min(1e-06)
        attack_fraction = attack_mask.float().mean().clamp_min(1e-06)
        sampled_fp = (normal_mask & (actions == 1)).float()
        sampled_fn = (attack_mask & (actions == 0)).float()
        fp_baseline = float(self.fpr_ema if self.fpr_ema is not None else 0.0)
        fn_baseline = float(self.fnr_ema if self.fnr_ema is not None else 0.0)
        fp_advantage = (
            normal_mask.float() * (sampled_fp - fp_baseline) / normal_fraction
        )
        fn_advantage = (
            attack_mask.float() * (sampled_fn - fn_baseline) / attack_fraction
        )
        if self.cost_adv_clip > 0.0:
            fp_advantage = fp_advantage.clamp(-self.cost_adv_clip, self.cost_adv_clip)
            fn_advantage = fn_advantage.clamp(-self.cost_adv_clip, self.cost_adv_clip)
        strength = self._warmup_strength(global_step)
        correction = strength * (
            self.lambda_fp * fp_advantage + self.lambda_fn * fn_advantage
        )
        constrained_advantage = task_advantages - correction
        with torch.no_grad():
            self.last_fp_adv_mean = float(fp_advantage.mean().item())
            self.last_fn_adv_mean = float(fn_advantage.mean().item())
            self.last_fp_adv_std = float(fp_advantage.std(unbiased=False).item())
            self.last_fn_adv_std = float(fn_advantage.std(unbiased=False).item())
            self.last_correction_mean = float(correction.mean().item())
            self.last_correction_std = float(correction.std(unbiased=False).item())
        return constrained_advantage

    def get_state(self) -> dict:
        return {
            "lambda_fp": float(self.lambda_fp),
            "lambda_fn": float(self.lambda_fn),
            "fpr_ema": float(self.fpr_ema or 0.0),
            "fnr_ema": float(self.fnr_ema or 0.0),
            "fp_violation": float(self.fp_violation),
            "fn_violation": float(self.fn_violation),
            "fp_adv_mean": float(self.last_fp_adv_mean),
            "fn_adv_mean": float(self.last_fn_adv_mean),
            "fp_adv_std": float(self.last_fp_adv_std),
            "fn_adv_std": float(self.last_fn_adv_std),
            "correction_mean": float(self.last_correction_mean),
            "correction_std": float(self.last_correction_std),
        }
