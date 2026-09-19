# jepa-risk-detection

日本語版: [`README.md`](README.md)

This repository contains Phase 1 experiments that use frozen V-JEPA 2 representations and masked-prediction error to study risk-labeled intervals in the Nexar Collision Prediction dataset.

The project is exploratory. The official Nexar test splits were not obtained, and the current A/B metrics reuse the internal validation split for model-selection or threshold selection. Do not interpret them as production safety results or future-accident prediction results.

## Two experimental paths

- **Path A — masked prediction error:** predict the feature representation of the final 16 frames from the first 48 frames, then compare the predicted and actual target representations. The resulting L1 error is tested for correlation with official risk intervals.
- **Path B — frozen features plus a linear probe:** average the V-JEPA 2 representation of all 64 observed frames and train only a linear classifier on Nexar risk labels.

Path A uses JEPA's prediction capability. Path B uses JEPA as a frozen feature extractor and does not use its prediction error. Neither path directly measures a large feature change between adjacent windows.

The full validation results are available in [`ab-result.en.md`](ab-result.en.md) and [`ab-result.md`](ab-result.md). The current headline results are:

| Path | ROC-AUC | Average Precision |
|---|---:|---:|
| A | 0.633691 | 0.037674 |
| B | 0.801785 | 0.144671 |

Both paths were evaluated on 321,931 common valid frames from 300 internal validation videos. These are exploratory validation metrics, not official-test metrics.

## Environment

Python is managed with `uv`.

```sh
uv python pin 3.12
uv lock
uv sync --locked
uv run pytest -m "not gpu"
uv run python -m jepa_risk.cli preflight \
  --config configs/phase1.yaml \
  --output artifacts/preflight
```

`preflight.json` records the available Python/uv/PyTorch runtime, accelerator backend, dataset paths, `ffprobe`, `pyarrow`, and model configuration. The raw predictor checkpoint is not required for the preflight command.

## Reproduction workflow

`prepare` reads the local Nexar metadata and creates the manifest, fixed video split, and sliding windows. The project uses only the official Nexar `train` split. Positive risk intervals are represented as `[time_of_alert, time_of_event)`.

```sh
uv run python -m jepa_risk.cli prepare \
  --config configs/phase1.yaml \
  --output artifacts/prepare-nexar-v2
```

After the local dataset and the Hugging Face encoder are available, extract frozen features and train/score the B probe:

```sh
uv run python -m jepa_risk.cli extract \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare-nexar-v2 \
  --output artifacts/features-p6-full \
  --split train \
  --selection all \
  --batch-size 4 \
  --local-files-only

uv run python -m jepa_risk.cli train-probe \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare-nexar-v2 \
  --features artifacts/features-p6-full \
  --output artifacts/probe-p6-full \
  --train-selection all \
  --validation-selection all

uv run python -m jepa_risk.cli score-probe \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare-nexar-v2 \
  --features artifacts/features-p6-full \
  --model artifacts/probe-p6-full/probe.pkl \
  --output artifacts/scores-p6-full \
  --split validation \
  --selection all
```

Path A uses the Meta predictor checkpoint configured in `configs/phase1.yaml`. The command supports video-level resume checkpoints for long runs:

```sh
uv run python -m jepa_risk.cli score-prediction \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare-nexar-v2 \
  --output artifacts/prediction-p6-validation \
  --split validation \
  --selection all \
  --batch-size 1 \
  --resume \
  --local-files-only
```

Evaluate saved A/B scores on the common frame contract:

```sh
uv run python -m jepa_risk.cli evaluate \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare-nexar-v2 \
  --probe-scores artifacts/scores-p6-full/scores.json \
  --prediction-scores artifacts/prediction-p6-validation/scores.json \
  --output artifacts/evaluation-p6-full-ab \
  --split validation \
  --selection all \
  --bootstrap 1000 \
  --seed 42
```

Generate the human-readable report and runtime estimates:

```sh
uv run python -m jepa_risk.cli report \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare-nexar-v2 \
  --output artifacts/report-p6-full-ab \
  --feature-measurement artifacts/features-p6-full/extraction-measurement.json \
  --prediction-measurement artifacts/prediction-p6-validation/scores.json \
  --probe-metrics artifacts/evaluation-p6-full-ab/metrics.json \
  --prediction-metrics artifacts/evaluation-p6-full-ab/metrics.json \
  --probe-scores artifacts/scores-p6-full/scores.json \
  --prediction-scores artifacts/prediction-p6-validation/scores.json
```

## Dataset and model inputs

The Nexar dataset must be downloaded separately from the [official Hugging Face dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction). This repository does not include videos, Parquet metadata, frames, or derived data. Use only the official `train` split for the current Phase 1 configuration.

```sh
uvx --from huggingface_hub hf download \
  nexar-ai/nexar_collision_prediction \
  --repo-type dataset \
  --include 'train/**' \
  --local-dir artifacts/data/nexar_train \
  --max-workers 4

curl -L \
  'https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet' \
  -o artifacts/data/nexar_train/metadata/train-0000.parquet
```

Review the [official dataset card](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/README.md) and [license](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE) before downloading or publishing derived results.

The B encoder is `facebook/vjepa2-vitl-fpc64-256` at the pinned revision in [`configs/phase1.yaml`](configs/phase1.yaml). The A path uses the raw `vitl.pt` checkpoint from the official Meta V-JEPA 2 release. The encoder model, raw checkpoint, and this project's code are separate licensing subjects.

## Publication and licensing

Before publication, read both the [English publication policy](PUBLICATION_POLICY.en.md) and the [Japanese publication policy](PUBLICATION_POLICY.md). The initial release excludes raw data, frames, caches, per-video scores, preview images, cookies, raw model checkpoints, and the trained B probe. A future B-probe release would require the separate derived-model rights check described in the policy.

The project code is released under the [MIT License](LICENSE), with `daikubo-jb` as the copyright holder. This license applies only to the project's own code. It does not override external dataset, model, checkpoint, or dependency terms. Third-party conditions are summarized in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The generated `probe.pkl` is a local pickle artifact for the experiment and is not included in the initial public repository. If the B classifier is cleared for release, it will be exported separately in a safer format such as JSON or NPZ, without training data, feature caches, or per-video scores, and accompanied by a model card.
