# R3KAN-IDS

Core implementation of **R3KAN-IDS: Risk-Aware Network Intrusion Detection under Scarce Attack Supervision and Adversarial Perturbations**.

The model uses a KAN actor and an independent MLP critic. Training combines Simple Policy Optimization (SPO), Risk-Constrained Advantage (RCA), and Risk-Aware Policy Consistency (RAPC).

## Installation

Python 3.10 or later is required. Use a separate virtual environment and install a PyTorch build appropriate for your CPU or CUDA environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Obtain the datasets from their providers and place the raw files in the locations below, or specify the input path when running preprocessing. Dataset files and trained weights are not distributed in this repository.

| Dataset | Default raw input |
| --- | --- |
| CICIDS2017 | `data/CICIDS2017/MachineLearningCVE/*.csv` |
| Edge-IIoTset | `data/Edge-IIoTset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv` |
| CIC IoT-DIAD 2024 | CSV files under `data/CIC-IoT-DIAD-2024/` |

```bash
python scripts/preprocess_cicids2017_clean.py
python scripts/preprocess_edge_iiotset_clean.py
python scripts/preprocess_cic_iot_diad_2024.py
```

Run `python scripts/<script_name>.py --help` for input and output options. Each script creates a fixed training/validation/test cache. Standardization is fitted on the training split only. Training reads these caches directly and does not generate substitute data when a cache is missing.

## Training

Run commands from the repository root. The three configurations use the same core architecture and training settings.

```bash
python train.py --config configs/cicids2017.yaml --seed 42
python train.py --config configs/edge_iiotset.yaml --seed 42
python train.py --config configs/diad2024.yaml --seed 42
```

Select the component configuration with `--variant`:

| Variant | Components |
| --- | --- |
| `base` | KAN actor, independent MLP critic, SPO |
| `rca` | Base + RCA |
| `full` | Base + RCA + RAPC (default) |

Clean experiments select checkpoints by clean validation F1. For robustness experiments, select by validation PGD-10 F1 at epsilon 0.02:

```bash
python train.py --config configs/cicids2017.yaml --selection pgd10_f1 --seed 42
```

For scarce-supervision settings, `--attack-ratio` specifies the Attack proportion in the **resulting training set**. All Normal training samples are retained. Validation and test data remain unchanged.

```bash
python train.py --config configs/cicids2017.yaml --variant rca --attack-ratio 0.02 --seed 42
```

RAPC uses `max(P,V)` by default. `--sampling p`, `--sampling v`, and `--sampling random` change only the within-class selection score. The class quotas, perturbation radius, and equal consistency weights remain fixed.

Use `--device cpu` for CPU execution, `--timesteps` to change the sample budget, and `--output-dir` to choose a fresh output directory. Training prints the location of `best.pt`. Existing run directories are not overwritten.

## Evaluation

Pass the saved checkpoint path:

```bash
python evaluate.py --checkpoint path/to/best.pt --attack clean
python evaluate.py --checkpoint path/to/best.pt --attack all
```

Clean-only evaluation uses the complete test split by default. Adversarial evaluation uses a fixed stratified subset of at most 50,000 test samples; its clean and attacked metrics use the same subset and checkpoint. `--attack all` evaluates clean inputs, FGSM, and PGD-10 at epsilon 0.02 and 0.05. Metrics are reported as fractions rather than percentages.

```bash
python evaluate.py --checkpoint path/to/best.pt --attack pgd10 --epsilon 0.02 --max-samples 50000 --subset-seed 3407
```

An optional `--output` argument saves the evaluation output as JSON.

## Implementation

| File | Responsibility |
| --- | --- |
| `agent.py`, `kan_layers.py` | KAN actor and independent value critic |
| `rca.py` | Greedy-action risk feedback and sampled-action advantage correction |
| `rapc.py` | P/V scoring, bilateral quotas, PGD-3, and detached-target consistency |
| `trainer.py` | Mini-batch SPO, value loss, entropy, and RAPC optimization |
| `datasets.py`, `environment.py` | Fixed data splits and single-step detection decisions |
| `adversarial_constraints.py`, `evaluation.py` | Constrained feature-space attacks and detection metrics |
| `train.py`, `evaluate.py` | Training and evaluation entry points |

RCA modifies actor advantages before mini-batch normalization. The critic continues to regress the immediate task reward. RAPC uses a fixed 25% exposure ratio, equal initial class quotas with unused-slot transfer, and equal KL weights. Its clean policy targets and adversarial inputs are cached before the optimization epochs.

Training PGD enforces the L-infinity radius only. Test-time attacks additionally enforce training-derived feature ranges and freeze features with two training-set values. PGD-10 starts from clean inputs without random restarts. Checkpoint selection uses validation data only.

The KAN backbone follows the learnable edge-function formulation described in [Liu et al., KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756).
