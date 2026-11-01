import argparse
import json
from pathlib import Path
import torch
from agent import create_agent
from config import make_config
from datasets import load_splits
from evaluation import evaluate, stratified_subset
from adversarial_constraints import infer_feature_constraints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attack", choices=["clean", "fgsm", "pgd10", "all"], default="clean"
    )
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--subset-seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.epsilon <= 0 or args.batch_size < 1:
        parser.error("--epsilon and --batch-size must be positive")
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = make_config(checkpoint["config"])
    data = load_splits(cfg.dataset, args.data_dir or cfg.data_dir)
    if checkpoint["feature_dim"] != data["X_train"].shape[1]:
        raise ValueError("Checkpoint and data feature dimensions differ")
    agent = create_agent(cfg, checkpoint["feature_dim"]).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"], strict=True)
    cap = (
        args.max_samples
        if args.max_samples is not None
        else 0 if args.attack == "clean" else 50000
    )
    indices = stratified_subset(data["y_test"], cap, args.subset_seed)
    x, y = (data["X_test"][indices], data["y_test"][indices])
    constraints = (
        infer_feature_constraints(data["X_train"]) if args.attack != "clean" else None
    )
    settings = [("clean", {})]
    if args.attack == "all":
        for epsilon in [0.02, 0.05]:
            settings.extend(
                [
                    (
                        f"fgsm_{epsilon:g}",
                        dict(
                            epsilon=epsilon,
                            steps=1,
                            step_size=epsilon,
                            constraints=constraints,
                        ),
                    ),
                    (
                        f"pgd10_{epsilon:g}",
                        dict(
                            epsilon=epsilon,
                            steps=10,
                            step_size=epsilon / 5,
                            constraints=constraints,
                        ),
                    ),
                ]
            )
    elif args.attack != "clean":
        steps = 1 if args.attack == "fgsm" else 10
        settings.append(
            (
                f"{args.attack}_{args.epsilon:g}",
                dict(
                    epsilon=args.epsilon,
                    steps=steps,
                    step_size=args.epsilon if steps == 1 else args.epsilon / 5,
                    constraints=constraints,
                ),
            )
        )
    result = dict(
        dataset=cfg.dataset,
        checkpoint=str(Path(args.checkpoint)),
        samples=len(y),
        subset_seed=args.subset_seed,
        selection_metric=cfg.selection_metric,
        metrics={},
    )
    for name, options in settings:
        result["metrics"][name] = evaluate(
            agent, x, y, device, args.batch_size, **options
        )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
