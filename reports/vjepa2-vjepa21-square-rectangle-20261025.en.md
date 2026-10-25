# V-JEPA 2 vs. V-JEPA 2.1 for Risk Classification: Input-Shape Comparison

[日本語版](vjepa2-vjepa21-square-rectangle-20261025.md)

## Summary

We compared frozen V-JEPA 2 and V-JEPA 2.1 encoders on the same fixed data split under four conditions: square inputs with black padding and landscape inputs without padding. A separate probe was fitted on training data for each condition and evaluated on the same 40 validation videos.

Validation window AP was 0.144516 for square V-JEPA 2 and 0.103784 for landscape V-JEPA 2; for V-JEPA 2.1, it was 0.057622 and 0.067239, respectively. The landscape-minus-square AP difference was -0.040731 for V-JEPA 2 and +0.009617 for V-JEPA 2.1. Both video-level paired-bootstrap 95% intervals include zero. This comparison did not show a consistent performance improvement from landscape inputs. Intervals that include zero do not prove that there is no difference or that the conditions are equivalent.

The V-JEPA 2 landscape condition changes the spatial token grid and positional coordinates. V-JEPA 2.1 keeps the total token count but increases the resolution of the active image. The observed differences therefore cannot be attributed to padding alone, and they also do not show that padding is irrelevant.

## Scope and data

This is a risk-window classification evaluation using internal splits created from the official `train` split of the Nexar Collision Prediction dataset. The official `test-public` and `test-private` splits were not used. The 200 videos were split by video into 160 training videos and 40 validation videos.

| Split | Videos | Windows | Positive windows | Videos with positive windows |
|---|---:|---:|---:|---:|
| Training | 160 | 11,390 | 254 | 81 |
| Validation | 40 | 2,905 | 65 | 19 |

All source videos were 1,280×720. Each window covered 2 seconds and 64 frames, with a 0.5-second stride. A window was labeled positive when its `available_at` timestamp fell within a risk interval; mere overlap with an interval was not sufficient. In validation, two of the 21 videos with risk annotations had short events that did not coincide with a decision timestamp, so they had no positive windows. Adjacent windows overlap by 75%, so the window count should not be interpreted as an independent sample count.

The 40-video validation set is a reused fixed pilot, not an untouched final holdout. Input dimensions, probe settings, and thresholds were not selected on validation for these conditions, but the effects of reusing the validation set cannot be ruled out.

## Models and four input conditions

V-JEPA 2 is a ViT-L/16 model with approximately 300M parameters. The model ID is `facebook/vjepa2-vitl-fpc64-256`, at revision [`b3c1679b7c34d3255ef3547f27c7b226aefab26f`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f). The comparison record says that the encoder branch used matched 388 tensors from the existing Hugging Face weights and was not the Hub-default `target_encoder` branch itself. The square baseline AP also matched its prior value. Neither check proves numerical equivalence of all outputs across implementation paths.

V-JEPA 2.1 is a ViT-L/16 model with approximately 300M parameters, an EMA student distilled from an approximately 2B-parameter ViT-G teacher. The checkpoint used was `vjepa2_1_vitl_dist_vitG_384.pt`; the Meta code was pinned to [revision `204698b45b3712590f06245fbfba32d3be539812`](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812).

| Condition | Model input | Active image | Spatial grid | Spatiotemporal tokens |
|---|---:|---:|---:|---:|
| V-JEPA 2, square | 256×256 | 256×144, padded by 56 px above and below | 16×16 | 8,192 |
| V-JEPA 2, landscape | 256×144 | 256×144, no padding | 9×16 | 4,608 |
| V-JEPA 2.1, square | 384×384 | 384×216, padded by 84 px above and below | 24×24 | 18,432 |
| V-JEPA 2.1, landscape | 512×288 | 512×288, no padding | 18×32 | 18,432 |

## Video preprocessing

The same videos, windows, and frame IDs were used in all conditions. RGB frames were processed with PIL BILINEAR resizing while preserving aspect ratio; there was no screen crop or aspect-ratio distortion. For square inputs, black pixels were added above and below before normalization. The black padding occupies 43.75% of each square input. Landscape conditions have no padding.

For V-JEPA 2, the active image is 256×144 in both shapes, but the spatial token grid changes from 16×16 to 9×16. For V-JEPA 2.1, the landscape active image has 4/3 the side length and 16/9 the area of the square condition. Its total token count stays the same, but the grid and image resolution differ. Neither comparison isolates padding as the only changed factor.

For all conditions, RGB values were rescaled to [0, 1] and normalized with ImageNet means and standard deviations. Additional processor resizing and center cropping were disabled. This evaluation-specific letterboxing is not a required official model procedure and differs from the official evaluation preprocessing.

## Probe and evaluation

The encoders were frozen. Features were computed by averaging the final normalized tokens across time, height, and width. The same probe procedure was fitted independently for each condition:

- `StandardScaler` and class-balanced `LogisticRegression`, fitted on training data only
- `C=0.01`, `lbfgs`, seed 42, maximum 2,000 iterations
- Primary metric: window AP (Average Precision)
- AP differences: video-level paired bootstrap, 1,000 resamples, seed 42, percentile 95% intervals

AP measures whether positive examples are ranked above negative examples; it is not accuracy. Window AP and frame AP use different evaluation units and should not be compared directly. Differences and intervals were calculated from unrounded values and are shown to six decimal places.

## Results

| Condition | Train window AP / ROC AUC | Validation window AP / ROC AUC | Validation frame AP / ROC AUC |
|---|---:|---:|---:|
| V-JEPA 2, square | 0.629291 / 0.991417 | 0.144516 / 0.794117 | 0.104294 / 0.786743 |
| V-JEPA 2, landscape | 0.626648 / 0.992211 | 0.103784 / 0.776376 | 0.073955 / 0.768049 |
| V-JEPA 2.1, square | 0.535919 / 0.986073 | 0.057622 / 0.732140 | 0.050938 / 0.719361 |
| V-JEPA 2.1, landscape | 0.539652 / 0.987757 | 0.067239 / 0.707855 | 0.059994 / 0.701584 |

The AP difference is the value for the right-hand condition minus the left-hand condition.

| Comparison | AP difference | 95% percentile interval |
|---|---:|---:|
| V-JEPA 2, landscape minus square | -0.040731 | [-0.171811, 0.008179] |
| V-JEPA 2.1, landscape minus square | +0.009617 | [-0.081012, 0.063030] |
| Square: V-JEPA 2.1 minus V-JEPA 2 | -0.086894 | [-0.253621, 0.110627] |
| Landscape: V-JEPA 2.1 minus V-JEPA 2 | -0.036546 | [-0.127183, 0.070667] |

All four conditions scored 2,905/2,905 validation windows (100% coverage). Frame coverage was 43,446/45,859 (94.738%), with 968 positive valid frames. There were no evaluation failures or probe convergence warnings.

## Interpretation and limitations

- On this fixed validation set, the window AP point estimate decreased for V-JEPA 2 and increased for V-JEPA 2.1 under landscape input. Both 95% intervals include zero, so the direction and size of either change are uncertain.
- Frame AP moved in the same model-specific directions as window AP, while frame ROC AUC decreased for both models. The metrics do not show a uniform change.
- The model-to-model differences under both square and landscape inputs also have 95% intervals that include zero. The results do not establish model superiority or equivalence.
- The V-JEPA 2 landscape condition changes its token grid and positional coordinates; the V-JEPA 2.1 landscape condition increases active-image resolution. The causal effect of padding alone cannot be estimated from this comparison.
- The models differ in pretraining data and recipe, distillation, resolution, weights, and implementation path. Using common downstream training and validation videos does not isolate the effect of any one factor.
- This is an exploratory risk-window classification study. It does not establish a general model ranking, advance collision prediction, or safety in real driving.

## Reproducibility and release scope

This report records aggregate results, conditions, model identifiers, and evaluation methods. It does not include videos, frames, per-video scores, feature caches, probes, or model weights. The report alone is not a reproduction package. An exact reproduction requires lawful access to the Nexar data, the specified model revisions and checkpoint from their official providers, and compatible preprocessing, encoder, probe, and aggregation implementations. The official test splits, an unused independent holdout, inference speed, memory use, and safety performance were not evaluated.

## Dataset source, citation, and terms

The data were derived from the `train` split of [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction). Include Nexar's specified dataset citation and the paper referenced by the dataset card:

> Moura, Daniel C., and Zvitia, Orly. “Nexar Collison Dataset.” Hugging Face, 2025. <https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction>
>
> Daniel C. Moura, Shizhan Zhu, and Orly Zvitia. “Nexar Dashcam Collision Prediction Dataset and Challenge.” 2025. <https://arxiv.org/abs/2503.03848>

The dataset card identifies the Nexar Open Data License and requires attribution and citation. Selling, sublicensing, or redistributing the dataset for profit requires Nexar’s prior written consent. The license also prohibits malicious systems, unsafe driving, deepfakes or misinformation, re-identification or privacy violations, weaponization, exploitative uses involving accident-prone areas or individuals, and unlawful purposes. The current dataset card also states that accessing the dataset entails consent to share contact information; do not request access if you do not consent. See the [license revision reviewed for this project](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/ed1ffff7c2dfb9014ff476f079ba5acefc71c72c/LICENSE). Before obtaining data or publishing, review the latest [official license](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE) and [dataset card](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/README.md). Dataset terms are separate from model and code licenses.

For model documentation, see the [Meta V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2), the [V-JEPA 2.1 paper](https://arxiv.org/html/2603.14482v1), and the [Transformers V-JEPA 2 documentation](https://huggingface.co/docs/transformers/model_doc/vjepa2). Model files are not included in this report.
