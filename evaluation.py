import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from adversarial_constraints import project_feature_space
from project_metrics import strict_auprc


def stratified_subset(labels, max_samples, seed):
    if max_samples <= 0 or max_samples >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    parts = []
    for label in [0, 1]:
        indices = np.flatnonzero(labels == label)
        count = int(round(max_samples * len(indices) / len(labels)))
        parts.append(rng.choice(indices, size=min(count, len(indices)), replace=False))
    result = np.concatenate(parts)
    rng.shuffle(result)
    return result


def attack_batch(agent, x, y, epsilon, steps, step_size, constraints=None):
    original = x.detach()
    adversarial = original.clone()
    for _ in range(max(int(steps), 1)):
        adversarial.requires_grad_(True)
        logits, _ = agent(adversarial)
        loss = F.cross_entropy(logits, y)
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        candidate = adversarial.detach() + float(step_size) * gradient.sign()
        adversarial = project_feature_space(
            original,
            candidate,
            epsilon,
            feature_min=(
                torch.as_tensor(constraints.minimum, device=x.device)
                if constraints is not None
                else None
            ),
            feature_max=(
                torch.as_tensor(constraints.maximum, device=x.device)
                if constraints is not None
                else None
            ),
            binary_mask=(
                torch.as_tensor(constraints.binary_mask, device=x.device)
                if constraints is not None
                else None
            ),
        )
    return adversarial


def evaluate(
    agent,
    x,
    y,
    device,
    batch_size=512,
    epsilon=0.0,
    steps=0,
    step_size=0.0,
    constraints=None,
):
    was_training = agent.training
    agent.eval()
    probabilities = []
    try:
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(
                x[start : start + batch_size], dtype=torch.float32, device=device
            )
            yb = torch.as_tensor(
                y[start : start + batch_size], dtype=torch.long, device=device
            )
            if epsilon > 0:
                xb = attack_batch(agent, xb, yb, epsilon, steps, step_size, constraints)
            with torch.no_grad():
                logits, _ = agent(xb)
                probabilities.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    finally:
        agent.train(was_training)
    score = np.concatenate(probabilities)
    prediction = (score >= 0.5).astype(np.int64)
    fp = int(((prediction == 1) & (y == 0)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    tp = int(((prediction == 1) & (y == 1)).sum())
    return dict(
        accuracy=float(accuracy_score(y, prediction)),
        f1=float(f1_score(y, prediction, zero_division=0)),
        fpr=fp / max(fp + tn, 1),
        fnr=fn / max(fn + tp, 1),
        mcc=float(matthews_corrcoef(y, prediction)),
        auprc=strict_auprc(y, score),
        pauc_fpr_0p01=float(roc_auc_score(y, score, max_fpr=0.01)),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        samples=len(y),
    )
