# Evaluating V-JEPA 2 with probes refit for lower-detail input conditions

[日本語版](jepa-low-resolution-probe-refit-20261016.md)

GitHub versions: [Japanese report](https://github.com/daikubo-jb/jepa-risk-detection/blob/main/reports/jepa-low-resolution-probe-refit-20261016.md) / [English report](https://github.com/daikubo-jb/jepa-risk-detection/blob/main/reports/jepa-low-resolution-probe-refit-20261016.en.md)

**Experiment:** A fixed pilot of 160 training videos and 40 validation videos selected from internal train/validation splits of Nexar Collision Prediction

**Summary data:** Three-condition comparison dated 2026-10-14

## Summary

We froze the V-JEPA 2 encoder—a model that converts video into numerical representations called features—meaning that its pretrained weights were not updated. We varied the spatial detail of the input video across three conditions, 256, 128, and 64, and refit only a linear probe, a small classifier that produces risk scores from features, with the same settings for each condition. Window-level Average Precision (AP) on the validation data, which were not used for fitting, was 0.144516 for the 256 condition, 0.107617 for 128, and 0.101968 for 64. A window is a time segment extracted from a video. Although the point estimates for 128 and 64 were lower than for 256, the 95% confidence intervals for the differences from a video-level paired bootstrap—resampling the same videos across conditions—both include zero. These intervals express uncertainty due to sampling variation. This pilot alone cannot establish the direction or magnitude of the differences.

AP evaluates how well the positive examples, here risk-labeled segments, are ranked near the top of the scores, using precision (the fraction of predicted positives that are truly positive) and recall (the fraction of actual positives that are retrieved). It is distinct from accuracy: AP=0.144516 does not mean that approximately 14% of predictions are correct.

This experiment reduces input detail. Images downsampled to 128/64 are resized back up to 256 before entering the encoder, so the encoder input size and token grid are the same in every condition. This is not an experiment measuring reductions in inference time or memory usage.

## Terms and evaluation metrics

| Term | Meaning in this report |
|---|---|
| Encoder, features, freezing | An encoder converts video into numerical representations called features. Freezing means leaving pretrained weights unchanged; the same V-JEPA 2 is used in every condition |
| Linear probe | A small classifier that weights features to produce a risk score. Here, StandardScaler standardizes the features, and LogisticRegression is fitted. Both are fitted using only the training split. Only this probe pipeline is refit for each condition, not V-JEPA 2 itself |
| Precision / recall | Precision is the fraction of predicted positives that are truly positive; recall is the fraction of actual positives that are retrieved. A positive here is an evaluation item with a risk-segment label |
| AP / PR-AUC | AP (Average Precision) is a ranking metric that sums precision values across thresholds, weighted by increments in recall. Computing PR-AUC (area under the precision-recall curve) by trapezoidal integration linearly interpolates between points. This differs from AP and need not yield the same value. The table reports AP |
| Positive prevalence and evaluation set | Positive prevalence is the fraction of evaluation items that are positive. AP depends on prevalence and the composition of the evaluation set. Comparisons should use the same evaluation set, labels, and unit; AP values from different datasets should not be compared directly |
| Window-level / frame-level evaluation | A window is an extracted time segment; a frame is one image in a video. Window AP is computed from scores and labels for windows, while frame AP uses scores and labels for valid frame rows. The units and positive prevalence differ, so the relative magnitudes do not directly indicate a performance difference |
| Paired bootstrap and 95% confidence interval | Validation videos are sampled with replacement at the video level, allowing repeated selections. Each resample uses the same videos across conditions to compute the AP difference. The 2.5th and 97.5th percentiles of the distribution of differences from 1,000 resamples form the 95% percentile interval. Windows are not resampled independently. This estimates uncertainty due to sampling variation. It neither assigns a 95% probability that the true difference lies in this particular interval nor proves equivalence when the interval includes zero |
| Train / validation and pilot | Training data are used to fit the probe; validation data are not used for fitting. A pilot is a small exploratory evaluation, here using a fixed set of 40 videos |
| Stride, letterbox, token grid | Stride is the interval by which a time window advances. Letterboxing preserves the image aspect ratio and adds padding to fit a specified size. The token grid is the arrangement of image regions processed by the encoder; both it and the input size remain fixed across the downsample-and-upsample conditions |
| Threshold, event detection, false alarms | A threshold is the score boundary used to decide whether a detection occurs. Event detections and false alarms count detections matched against events at that threshold. They are separate from AP, which evaluates the ranking of scores |

Terminology references: see the [scikit-learn metrics explanation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) for AP, precision, recall, and the difference from trapezoidal integration; the [precision-recall curve specification](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) for the relationship to positive prevalence; and the [SciPy bootstrap explanation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) for a general description of sampling with replacement and percentile intervals. These are references for definitions, not claims about this experiment's implementation or execution libraries.

## Experimental questions

This comparison addresses two questions.

1. How does AP for ranking risk-segment windows change when the spatial detail of input images is reduced while using the same frozen encoder?
2. With a probe refit for each detail condition in this experiment, how do AP values compare across the conditions?

The variable being changed is the spatial information retained in the input image. Neither the weights of V-JEPA 2 itself nor the temporal windowing is changed.

## Experimental conditions

| Item | Condition |
|---|---|
| Dataset | Internal splits made from the official Nexar Collision Prediction `train` split. The official test split was not used |
| Training | 160 videos, 11,390 windows (254 positive, 11,136 negative) |
| Validation | The same fixed pilot of 40 videos, 2,905 windows |
| Split overlap | 0 videos |
| Model | `facebook/vjepa2-vitl-fpc64-256`, revision [`b3c1679b7c34d3255ef3547f27c7b226aefab26f`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f) |
| Time window | 2 seconds, 64 frames, 0.5-second stride |
| Encoder | Frozen and identical across conditions |
| Probe | Both StandardScaler and class-balanced LogisticRegression are fitted using only the training split. C=0.01, lbfgs, seed=42 |
| Input processing | Letterbox to 256×256, then create the 256/128/64 detail conditions. For 128/64, downsample with BILINEAR and resize back up to 256×256 before passing the image to the same encoder |
| Main evaluation | Window-level AP. Intervals for differences use a validation-video-level paired bootstrap with 1,000 resamples and seed=42 |

All 2,905 validation windows were scored in every condition. There were 0 feature-extraction failures in the 128/64 conditions. For the frame metrics, 43,446 of 45,859 rows (94.74%) were valid in each condition. All 21 events could be scored over their full intervals.

## Results

| Detail condition | Window AP | Difference from 256 (95% bootstrap interval) | Frame AP | Event detections at a fixed threshold | False alarms |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.144516 | — | 0.104294 | 4 / 21 | 20 |
| 128 | 0.107617 | -0.036898 `[-0.184652, 0.083747]` | 0.082793 | 4 / 21 | 25 |
| 64 | 0.101968 | -0.042548 `[-0.199979, 0.072646]` | 0.083509 | 2 / 21 | 18 |

The event columns are reference values using a threshold of 0.8996675 shared across the three conditions. Because the history of how this threshold was selected was not checked in this work, these values are not treated as a comparison of operational performance.

The point estimates of the AP differences were negative for both 128 and 64, but both 95% intervals span zero. These results do not support a conclusion that lower detail caused a statistically significant deterioration. Although the probes were refit for the lower-detail conditions here, the conditions differ from earlier fixed-probe results obtained with a different training-set size. It is therefore also not possible to say that refitting recovered performance.

## Publication status and reproducibility

The `scripts/resolution_ablation.py` used for this comparison and the corresponding additions to `src/jepa_risk/preprocessing.py` have not yet been published. This report presents saved experimental results; cloning the currently published repository alone does not provide a reproducible version of this experiment.

The standard summary produced by this version of the comparison script aggregates frame-level metrics and differences. The window-level aggregation procedure needed to recalculate the table's window AP and the 95% confidence intervals for window AP differences is also an outstanding publication task. Publishing and validating the execution code and aggregation procedure require separate work.

## Reproduction notes

- See the project's [`README.md`](../README.md) and `uv.lock` for the Python 3.12 configuration, dependencies, and setup steps.
- The comparison implementation is the unpublished `scripts/resolution_ablation.py`. The preprocessing file is [`src/jepa_risk/preprocessing.py`](../src/jepa_risk/preprocessing.py), but the additional detail transformations required by this comparison have not yet been published.
- The execution mode is `--mode pilot --probe-mode refit --selection pilot`, the comparison conditions are 256/128/64, and bootstrap uses 1,000 resamples with seed 42. Use the model ID and revision fixed in [`configs/phase1.yaml`](../configs/phase1.yaml), together with 2 seconds / 64 frames / 0.5-second stride.
- Execution requires locally obtained Nexar data, prepared manifests, the 256 features and probe metadata used for matching, threshold information from the existing evaluation, model weights, and related inputs. These data, weights, feature caches, and per-video scores are not distributed with this document. See [`PUBLICATION_POLICY.md`](../PUBLICATION_POLICY.md) for the initial publication policy.
In addition to refitting the probe for path B, this comparison script also runs path A. It therefore requires existing path-A baseline scores and the official predictor checkpoint, as well as the probe file and metadata for the 256 condition. These files are not distributed with the documents.

- Specify an unused output location when rerunning. The implementation is designed not to overwrite existing outputs.

## Limitations of interpretation

- Only the fixed pilot of 40 videos was evaluated. This is not an evaluation on the official test split or on a large independent holdout.
- The bootstrap intervals for the AP differences include zero, so this comparison cannot establish a statistical difference.
- A probe is fitted on the 160 training videos for each of 256/128/64. These numbers cannot be compared directly with those from the earlier fixed probe trained on 1,200 videos. They are not evidence of recovery through refitting.
- The event metrics are reference comparisons at the same threshold, whose selection history has not been checked. Neither threshold reselection nor an evaluation of an operational threshold was performed.
- This is an exploratory evaluation that classifies risk-segment labels at the window level. It does not demonstrate the ability to predict collisions in advance or to function safely in real driving environments.
- Even when input detail is reduced, the encoder input size remains 256×256. Reductions in inference time, memory usage, or power consumption were not evaluated.

## Data, model, and licenses

The data used were Nexar Collision Prediction. Refer to the dataset citation specified by Nexar and to the paper. The dataset is not covered by this project's MIT License. Users should check the official license applicable when they obtain the data. No video or data derivatives are included in this deliverable.

- [Nexar Collision Prediction dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)
- [Nexar Open Data License (revision `ed1ffff7c2dfb9014ff476f079ba5acefc71c72c`)](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/ed1ffff7c2dfb9014ff476f079ba5acefc71c72c/LICENSE)
- [Nexar dataset paper, arXiv:2503.03848](https://arxiv.org/abs/2503.03848)
- [V-JEPA 2 model card at the configured revision](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f) — The Hugging Face page at this revision displays MIT.

The MIT License for this project's code does not determine the conditions governing external data or models, or whether derivatives such as probes may be redistributed. This report is limited to aggregate values and reproduction information. It does not distribute videos, frames, model weights, probes, features, or per-video outputs. The redistribution conditions for derived probes remain unverified.

## Citations

> Moura, Daniel C., and Zvitia, Orly. “Nexar Collison Dataset.” Hugging Face, 2025. <https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction>

> Daniel C. Moura, Shizhan Zhu, and Orly Zvitia. “Nexar Dashcam Collision Prediction Dataset and Challenge.” 2025. <https://arxiv.org/abs/2503.03848>
