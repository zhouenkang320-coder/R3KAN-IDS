import math
import torch
import torch.nn.functional as F
from adversarial_constraints import project_feature_space


def select_states(scores, labels, fraction):
    budget = (
        min(len(labels), max(1, math.ceil(fraction * len(labels))))
        if fraction > 0
        else 0
    )
    normal = torch.nonzero(labels == 0, as_tuple=False).flatten()
    attack = torch.nonzero(labels == 1, as_tuple=False).flatten()
    if len(normal) == 0:
        normal_budget, attack_budget = (0, min(budget, len(attack)))
    elif len(attack) == 0:
        normal_budget, attack_budget = (min(budget, len(normal)), 0)
    else:
        normal_budget = int(round(budget / 2))
        attack_budget = budget - normal_budget
        normal_budget = min(normal_budget, len(normal))
        attack_budget = min(attack_budget, len(attack))
        remaining = budget - normal_budget - attack_budget
        extra = min(remaining, len(normal) - normal_budget)
        normal_budget += extra
        remaining -= extra
        attack_budget += min(remaining, len(attack) - attack_budget)
    selected = torch.zeros_like(labels, dtype=torch.bool)
    for indices, count in [(normal, normal_budget), (attack, attack_budget)]:
        if count:
            selected[indices[torch.topk(scores[indices], k=count).indices]] = True
    return selected


def prepare_batch(agent, states, labels, config):
    probe = states.detach().clone().requires_grad_(True)
    logits, _ = agent(probe)
    probabilities = torch.softmax(logits, dim=-1)
    p = torch.where(labels == 1, probabilities[:, 0], probabilities[:, 1]).detach()
    margin = logits.gather(1, labels[:, None]).squeeze(1)
    margin = margin - logits.gather(1, (1 - labels)[:, None]).squeeze(1)
    gradient = torch.autograd.grad(margin.sum(), probe, only_inputs=True)[0]
    distance = torch.relu(margin.detach()) / gradient.detach().abs().sum(1).clamp_min(
        1e-12
    )
    v = torch.clamp(1 - distance / config.rapc_epsilon, 0, 1)
    if config.rapc_local_risk_mode == "p":
        score = p
    elif config.rapc_local_risk_mode == "v":
        score = v
    elif config.rapc_local_risk_mode == "random":
        score = torch.rand_like(p)
    else:
        score = torch.maximum(p, v)
    selected = select_states(score.detach(), labels, config.rapc_fraction)
    weights = selected.float()
    original = states.detach()
    adversarial = original.clone()
    radius = torch.full_like(score, float(config.rapc_epsilon)).view(-1, 1)
    mask = weights[:, None]
    if selected.any():
        perturbation = torch.empty_like(original).uniform_(-1, 1) * radius
        adversarial = project_feature_space(
            original, original + perturbation * mask, radius
        )
        step_size = radius if config.rapc_inner_steps == 1 else radius / 2
        for _ in range(config.rapc_inner_steps):
            current = adversarial.detach().requires_grad_(True)
            current_logits, _ = agent(current)
            losses = F.cross_entropy(current_logits, labels, reduction="none")
            objective = (losses * weights).sum() / weights.sum().clamp_min(1)
            gradient = torch.autograd.grad(objective, current, only_inputs=True)[0]
            candidate = current.detach() + step_size * gradient.sign() * mask
            adversarial = project_feature_space(original, candidate, radius)
    return (adversarial.detach(), probabilities.detach(), weights.detach())


def consistency_loss(agent, states, targets, weights):
    if not weights.sum().item():
        return states.new_zeros(())
    logits, _ = agent(states)
    divergence = F.kl_div(F.log_softmax(logits, dim=-1), targets, reduction="none").sum(
        -1
    )
    return (weights * divergence).sum() / (weights.sum() + 1e-08)
