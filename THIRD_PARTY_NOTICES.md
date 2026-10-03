# Third-Party Notices

This file records external components used by the experiment. The project's own source code is licensed under the root [MIT License](LICENSE). This file is not a license for any external data, model, checkpoint, or dependency.

## Nexar Collision Prediction dataset

Phase 1 uses only the official `train` split of the [Nexar Collision Prediction dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction). The dataset is governed by the [Nexar Open Data License](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE), not by this repository's MIT License. The initial release does not contain the dataset or derived per-video artifacts.

Required citation:

> Moura, Daniel C., and Zvitia, Orly. “Nexar Collison Dataset.” Hugging Face, 2025, https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction.

The dataset card also requests citation of the accompanying dataset paper:

> Daniel C. Moura, Shizhan Zhu, Orly Zvitia, “Nexar Dashcam Collision Prediction Dataset and Challenge,” arXiv:2503.03848, 2025, https://arxiv.org/abs/2503.03848.

## V-JEPA 2 Hugging Face encoder

The B path retrieves `facebook/vjepa2-vitl-fpc64-256` at revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`. The pinned Hugging Face page displays an MIT license: [model page](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f).

The model files are not bundled in this repository.

## Meta V-JEPA 2 implementation

The A path follows the Meta V-JEPA 2 architecture and is pinned to commit `204698b45b3712590f06245fbfba32d3be539812`. The official README states that most of the repository is MIT and identifies these Apache-2.0 files:

- `src/datasets/utils/video/randaugment.py`
- `src/datasets/utils/video/randerase.py`
- `src/datasets/utils/worker_init_fn.py`

Official source: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812).

`src/jepa_risk/predictor.py` implements a compatibility path around the Meta V-JEPA 2 architecture and checkpoint naming. It includes code whose operations and structure correspond to the pinned Meta implementation (including RoPE attention, Transformer blocks, video patch embedding, and predictor token handling), alongside project-specific changes. The extent of copying or adaptation is not asserted here. The file carries an attribution and MIT license reference; the complete pinned Meta MIT license is retained in [`THIRD_PARTY_LICENSES/META-VJEPA2-MIT.txt`](THIRD_PARTY_LICENSES/META-VJEPA2-MIT.txt). The current tree does not include the three Apache-2.0 utility files listed above. See the pinned [Meta source](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812).

## Meta raw predictor checkpoint

The local A-path `vitl.pt` has SHA-256 `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012` and size 5,127,726,842 bytes. The official Hugging Face file [`facebook/vjepa2-vitl-fpc64-256/original/model.pth`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/main/original/model.pth) displays the same SHA-256 and size (5.13 GB), and its model page displays MIT. This establishes byte identity with that official HF-hosted MIT-marked file; it does not establish which URL was used for the historical download. The project records `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt` as the Meta source URL, but no download log was found. The checkpoint is not redistributed or mirrored in this repository. See [`PUBLICATION_POLICY.en.md`](PUBLICATION_POLICY.en.md) for the evidence and its limits.

## Python dependencies

Runtime dependencies are declared in [`pyproject.toml`](pyproject.toml) and resolved in [`uv.lock`](uv.lock). They are installed from their respective package sources and remain subject to their own licenses. The project does not copy their source into this repository.
