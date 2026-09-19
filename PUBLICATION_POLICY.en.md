---
title: Data, Model, and Publication Policy
date: 2026-09-19
status: publication-gate
---

# Data, Model, and Publication Policy

This document separates the rights and release scope for project code, external data, external models, checkpoints, and generated artifacts. It is an operational release checklist, not legal advice.

## Current decision

- Phase 1 uses only the `train` split of the Nexar Collision Prediction dataset. The official public and private test splits were not used.
- The initial repository release must not include MP4 files, Parquet metadata, extracted frames, feature caches, per-video scores, preview images, cookies, raw model checkpoint files, or the B probe. A future B-probe release would require a separate derived-model rights check.
- Model IDs, source repositories, pinned revisions/commits, download instructions, and verification hashes may be documented without redistributing the model files.
- The raw Meta predictor checkpoint `vitl.pt` is not redistributed or mirrored because its separate checkpoint terms were not conclusively identified in the reviewed official materials.
- Aggregate reports may be released only after removing video-level identifiers, absolute paths, images, secrets, and other information that could expose individuals or source records.
- The project's own code has a root MIT `LICENSE` with `daikubo-jb` as the copyright holder. For this release, the project-owned code is treated as individually owned code. This does not change the separate terms for external data, models, checkpoints, or dependencies.

## 1. Nexar Collision Prediction

Official sources checked on 2026-09-15:

- [Dataset page](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)
- [Dataset card](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/README.md)
- [Nexar Open Data License](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE)

The checked license revision was `ed1ffff7c2dfb9014ff476f079ba5acefc71c72c`. The dataset license requires attribution to Nexar and the specified citation, requires redistribution to retain the copyright notice, conditions, and disclaimers, and prohibits selling, sublicensing, or otherwise redistributing the dataset for profit without prior written consent from Nexar.

The license also restricts malicious systems, unsafe driving or simulated unsafe behavior, deepfakes and misinformation, re-identification and privacy violations, weaponization, exploitative practices involving accident-prone regions or individuals, and unlawful use. Users must review the current official license themselves at download and release time.

Required citation:

> Moura, Daniel C., and Zvitia, Orly. “Nexar Collison Dataset.” Hugging Face, 2025, https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction.

This repository does not sublicense the Nexar dataset or grant downstream users dataset rights. Raw Nexar data, source annotations, extracted frames, detailed manifests, thumbnails, and per-video results are excluded from the initial release.

## 2. V-JEPA 2 encoder

Path B uses:

- Model: `facebook/vjepa2-vitl-fpc64-256`
- Pinned revision: `b3c1679b7c34d3255ef3547f27c7b226aefab26f`
- [Pinned model page](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f)

The pinned Hugging Face model page displays an MIT license. The model is not copied into this repository; users retrieve it from the official provider under the terms shown there.

## 3. Meta V-JEPA 2 code and A checkpoint

The A implementation is pinned to Meta repository commit `204698b45b3712590f06245fbfba32d3be539812` in [`configs/phase1.yaml`](configs/phase1.yaml). The official V-JEPA 2 README says that most of the repository is MIT, while these files are Apache-2.0:

- `src/datasets/utils/video/randaugment.py`
- `src/datasets/utils/video/randerase.py`
- `src/datasets/utils/worker_init_fn.py`

See the [official V-JEPA 2 README](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md) and [repository](https://github.com/facebookresearch/vjepa2). The current technical review classifies `src/jepa_risk/predictor.py` as a project-owned compatibility implementation rather than a vendored Meta source file. The current tree does not include the three Apache-2.0 utility files listed above. The Meta attribution and pinned commit remain documented; if Meta source is copied or adapted in the future, the applicable copyright and MIT/Apache-2.0 notices must be retained.

The A raw checkpoint is:

- URL: `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt`
- Current local SHA-256: `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012`
- Current local size: `5,127,726,842` bytes

The URL is listed by the official Meta README, but the reviewed official materials did not identify a separate license for this raw `.pt` file. The repository therefore does not distribute, mirror, or promise commercial redistribution rights for it. Recompute the hash and size immediately before publication.

## 4. Project code and dependencies

The project's own code is released under the root [MIT License](LICENSE), with `daikubo-jb` as the copyright holder. For this release, the project-owned code is treated as individually owned code. The MIT license applies only to project-owned code.

The project's MIT license does not override the Nexar dataset license, the V-JEPA 2 model terms, the raw checkpoint terms, or the licenses of declared Python dependencies. The project does not bundle those external assets in the initial source release.

## 5. Release artifact scope

Allowed candidates, subject to final inspection:

1. Source code, configuration templates, and CPU/synthetic tests.
2. Official source links, model IDs, pinned revisions, commits, download instructions, and verification hashes.
3. Aggregate reports that contain no video-level identifiers, raw-derived files, absolute paths, or secrets.
4. A future B linear-probe release, only if its derived-model redistribution is cleared, distributed without training data, feature caches, or per-video scores and accompanied by a model card.

Excluded from the initial release:

- Nexar videos, Parquet files, frames, raw annotations, and detailed manifests.
- V-JEPA 2 model files, `vitl.pt`, feature caches, and compressed archives containing them. The B probe is also excluded from the initial release; its separate rights check is deferred to any future model release.
- Per-video scores, failure intervals, thumbnails, previews, cookies, tokens, and local absolute paths.

## 6. Publication gate

- [x] Confirm the stated copyright holder and add a root `LICENSE` for project code.
- [x] Add a final third-party notice and retain applicable Meta MIT/Apache notices.
- [x] Re-check the current Nexar license, dataset citation, and ethical restrictions.
- [x] Re-check the pinned Hugging Face model license and revision.
- [x] Re-check the `vitl.pt` source and separate terms; exclude it from the initial release because those terms are not conclusively identified.
- [x] Inspect the current Git history, public tree, wheels, sdists, and candidate archives. GitHub release attachments will be rechecked if/when they are created.
- [x] Confirm no raw data, frames, caches, checkpoints, cookies, absolute paths, or video-level identifiers are released.
- [x] Exclude the B probe from the initial release; defer its derived-model rights check, safe export, and model card to any future model release.
- [x] State clearly that official-test evaluation and production safety validation are not complete.
