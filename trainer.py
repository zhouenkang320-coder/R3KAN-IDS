import numpy as np
import torch
from rca import RiskConstrainedAdvantage
from rapc import prepare_batch, consistency_loss


class Trainer:

    def __init__(self, config, agent, optimizer):
        self.config = config
        self.agent = agent
        self.optimizer = optimizer
        self.rca = (
            RiskConstrainedAdvantage(
                target_fpr=config.rc_target_fpr,
                target_fnr=config.rc_target_fnr,
                dual_lr=config.rc_dual_lr,
                lambda_max=config.rc_lambda_max,
                ema_alpha=config.rc_ema_alpha,
                warmup_steps=config.rc_warmup_steps,
                cost_adv_clip=config.rc_cost_adv_clip,
            )
            if config.use_rca
            else None
        )

    def correct_advantages(self, states, labels, actions, advantages, step):
        if self.rca is None:
            return advantages
        with torch.no_grad():
            logits, _ = self.agent(states)
            feedback = logits.argmax(-1)
        return self.rca.compute(advantages, actions, labels, feedback, step)

    def update(
        self,
        step,
        states,
        labels,
        actions,
        old_log_probs,
        advantages,
        returns,
        old_values,
    ):
        cfg = self.config
        scale = (
            min(1.0, max(0.0, step / cfg.rapc_warmup_steps))
            if cfg.rapc_warmup_steps
            else 1.0
        )
        adversarial = targets = weights = None
        if cfg.use_rapc and scale > 0:
            adversarial, targets, weights = prepare_batch(
                self.agent, states, labels, cfg
            )
        indices = np.arange(len(states))
        totals = dict(
            policy_loss=0.0, value_loss=0.0, entropy=0.0, rapc_loss=0.0, approx_kl=0.0
        )
        updates = 0
        stopped = False
        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            last_kl = 0.0
            for start in range(0, len(states), cfg.minibatch_size):
                batch = indices[start : start + cfg.minibatch_size]
                _, log_probs, entropy, values = self.agent.get_action_and_value(
                    states[batch], actions[batch]
                )
                log_ratio = log_probs - old_log_probs[batch]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    last_kl = (ratio - 1 - log_ratio).mean().item()
                advantage = advantages[batch]
                if cfg.advantage_normalization:
                    advantage = (advantage - advantage.mean()) / (
                        advantage.std() + 1e-08
                    )
                policy = -(
                    advantage * ratio
                    - advantage.abs() * (ratio - 1).pow(2) / (2 * cfg.epsilon)
                ).mean()
                value_error = (values - returns[batch]).pow(2)
                clipped = old_values[batch] + torch.clamp(
                    values - old_values[batch],
                    -cfg.value_clip_epsilon,
                    cfg.value_clip_epsilon,
                )
                value = (
                    0.5
                    * torch.maximum(
                        value_error, (clipped - returns[batch]).pow(2)
                    ).mean()
                )
                entropy_mean = entropy.mean()
                loss = policy + cfg.c_1 * value - cfg.c_2 * entropy_mean
                robust = states.new_zeros(())
                if adversarial is not None and weights[batch].sum() > 0:
                    robust = (
                        cfg.rapc_coef
                        * scale
                        * consistency_loss(
                            self.agent,
                            adversarial[batch],
                            targets[batch],
                            weights[batch],
                        )
                    )
                    loss = loss + robust
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()
                for key, item in [
                    ("policy_loss", policy),
                    ("value_loss", value),
                    ("entropy", entropy_mean),
                    ("rapc_loss", robust),
                ]:
                    totals[key] += item.item()
                totals["approx_kl"] += last_kl
                updates += 1
            if cfg.target_kl is not None and last_kl > cfg.target_kl:
                stopped = True
                break
        result = {key: value / max(updates, 1) for key, value in totals.items()}
        result["updates"] = updates
        result["kl_stopped"] = stopped
        if self.rca is not None:
            result.update(self.rca.get_state())
        return result
