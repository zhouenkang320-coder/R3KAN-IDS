import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from agent import create_agent
from config import load_config
from datasets import load_splits, attack_subset_indices
from environment import IDSEnvironment
from evaluation import evaluate, stratified_subset
from adversarial_constraints import infer_feature_constraints
from trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=["base", "rca", "full"], default="full")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--attack-ratio", type=float)
    parser.add_argument("--selection", choices=["clean_f1", "pgd10_f1"])
    parser.add_argument("--sampling", choices=["max", "p", "v", "random"])
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for argument, field in [
        ("seed", "random_seed"),
        ("timesteps", "total_timesteps"),
        ("data_dir", "data_dir"),
        ("output_dir", "output_dir"),
        ("device", "device"),
        ("selection", "selection_metric"),
        ("sampling", "rapc_local_risk_mode"),
    ]:
        if getattr(args, argument) is not None:
            setattr(cfg, field, getattr(args, argument))
    cfg.use_rca = args.variant != "base"
    cfg.use_rapc = args.variant == "full"
    cfg.validate()
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
    set_seed(cfg.random_seed)
    device = torch.device(
        cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    data = load_splits(cfg.dataset, cfg.data_dir)
    x_train, y_train = (data["X_train"], data["y_train"])
    constraints = (
        infer_feature_constraints(x_train)
        if cfg.selection_metric == "pgd10_f1"
        else None
    )
    if args.attack_ratio is not None:
        indices = attack_subset_indices(
            y_train, data["attack_type_train"], args.attack_ratio, cfg.random_seed
        )
        x_train, y_train = (x_train[indices], y_train[indices])
    if len(y_train) < cfg.batch_size:
        raise ValueError(
            "Training set is smaller than batch_size; reduce batch_size in the configuration"
        )
    x_val, y_val = (data["X_val"], data["y_val"])
    if cfg.selection_metric == "pgd10_f1":
        subset = stratified_subset(
            y_val, cfg.max_validation_samples, cfg.evaluation_seed
        )
        x_val, y_val = (x_val[subset], y_val[subset])
    ratio_name = (
        "natural" if args.attack_ratio is None else f"attack_{args.attack_ratio:g}"
    )
    output = (
        Path(cfg.output_dir)
        / cfg.dataset
        / args.variant
        / ratio_name
        / cfg.selection_metric
        / f"seed_{cfg.random_seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a different --output-dir."
        )
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)
    env = IDSEnvironment(x_train, y_train)
    env.X, env.y = (env.X.to(device), env.y.to(device))
    agent = create_agent(cfg, x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    trainer = Trainer(cfg, agent, optimizer)
    set_seed(cfg.random_seed + 1000003)
    step, next_evaluation, best_score = (0, cfg.eval_interval, -1.0)
    while step < cfg.total_timesteps:
        batch = env.sample_batch(cfg.batch_size)
        states, labels = (batch["states"], batch["labels"])
        with torch.no_grad():
            actions, log_probs, _, values = agent.get_action_and_value(states)
            rewards = torch.where(actions == labels, 1.0, -1.0)
            advantages = rewards - values
        advantages = trainer.correct_advantages(
            states, labels, actions, advantages, step
        )
        training = trainer.update(
            step, states, labels, actions, log_probs, advantages, rewards, values
        )
        step += cfg.batch_size
        if step >= next_evaluation or step >= cfg.total_timesteps:
            while next_evaluation <= step:
                next_evaluation += cfg.eval_interval
            attack = (
                {}
                if cfg.selection_metric == "clean_f1"
                else dict(
                    epsilon=0.02, steps=10, step_size=0.004, constraints=constraints
                )
            )
            validation = evaluate(
                agent, x_val, y_val, device, cfg.eval_batch_size, **attack
            )
            if validation["f1"] > best_score:
                best_score = validation["f1"]
                torch.save(
                    dict(
                        model_state_dict=agent.state_dict(),
                        config=cfg.to_dict(),
                        feature_dim=x_train.shape[1],
                        step=step,
                        validation=validation,
                        variant=args.variant,
                        attack_ratio=args.attack_ratio,
                    ),
                    output / "best.pt",
                )
            print(
                json.dumps(
                    dict(
                        step=step,
                        validation_f1=validation["f1"],
                        best_validation_f1=best_score,
                        **training,
                    )
                ),
                flush=True,
            )
    print(f"Checkpoint: {output / 'best.pt'}")


if __name__ == "__main__":
    main()
