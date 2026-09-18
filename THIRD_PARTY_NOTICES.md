# Third-Party Notices

This file records external components used by the experiment. The project's own source code is licensed under the root [MIT License](LICENSE). This file is not a license for any external data, model, checkpoint, or dependency.

## Nexar Collision Prediction dataset

Phase 1 uses only the official `train` split of the [Nexar Collision Prediction dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction). The dataset is governed by the [Nexar Open Data License](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE), not by this repository's MIT License. The initial release does not contain the dataset or derived per-video artifacts.

Required citation:

> Moura, Daniel C., and Zvitia, Orly. “Nexar Collison Dataset.” Hugging Face, 2025, https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction.

## V-JEPA 2 Hugging Face encoder

The B path retrieves `facebook/vjepa2-vitl-fpc64-256` at revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`. The pinned Hugging Face page displays an MIT license: [model page](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f).

The model files are not bundled in this repository.

## Meta V-JEPA 2 implementation

The A path follows the Meta V-JEPA 2 architecture and is pinned to commit `204698b45b3712590f06245fbfba32d3be539812`. The official README states that most of the repository is MIT and identifies these Apache-2.0 files:

- `src/datasets/utils/video/randaugment.py`
- `src/datasets/utils/video/randerase.py`
- `src/datasets/utils/worker_init_fn.py`

Official source: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812).

The current technical review classifies `src/jepa_risk/predictor.py` as a project-owned compatibility implementation, not a vendored Meta source file. The current tree does not include the three Apache-2.0 utility files listed above. If Meta source is copied or adapted in a future revision, the applicable copyright and MIT/Apache-2.0 notices must be retained.

## Meta raw predictor checkpoint

The A path uses the separately downloaded `vitl.pt` checkpoint from `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt`. The checkpoint is not redistributed or mirrored because its separate terms were not conclusively identified in the reviewed official materials. Its current local hash and size are recorded in [`PUBLICATION_POLICY.en.md`](PUBLICATION_POLICY.en.md).

## Python dependencies

Runtime dependencies are declared in [`pyproject.toml`](pyproject.toml) and resolved in [`uv.lock`](uv.lock). They are installed from their respective package sources and remain subject to their own licenses. The project does not copy their source into this repository.
