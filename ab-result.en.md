---
title: V-JEPA 2 Paths A/B Full Validation Results
date: 2026-09-15
status: exploratory-validation-complete-official-test-pending
---

# V-JEPA 2 Paths A/B Full Validation Results

## Executive summary

We compared two ways of using frozen V-JEPA 2 features on 300 internal validation videos and 21,554 windows from the Nexar Collision Prediction training split. Both paths ranked risk-labeled frames above a constant-score baseline, but Path B was stronger under the current setup.

| Path | What it tests | ROC-AUC | Average Precision | Constant-score AP |
|---|---|---:|---:|---:|
| A | Whether prediction error is higher in risk intervals | 0.633691 | 0.037674 | 0.024011 |
| B | Whether frozen features can classify risk intervals | 0.801785 | 0.144671 | 0.024011 |

Both paths were evaluated on the same 321,931 common valid frames. B achieved about 6.03 times the constant-score AP and about 3.84 times A's AP. These are exploratory validation results, not evidence of accident anticipation, production safety, or generalization to unseen datasets.

## 1. What the two hypotheses are

The goal is to test whether a frozen, pretrained V-JEPA 2 can provide useful information for identifying risk-labeled intervals in Nexar videos. V-JEPA 2 itself is not fine-tuned in either path.

### Path A: masked prediction error

#### Hypothesis

If JEPA finds the next part of a video difficult to predict, that interval may be unusual and therefore more likely to be a risk interval.

#### Method

Each input is a two-second window containing 64 frames. The first 48 frames are used as context and the final 16 frames as the target.

1. The predictor estimates the target feature representation from the first 48 frames.
2. The actual final 16 frames are encoded to obtain the target representation.
3. We compute the mean L1 distance between the predicted and actual target representations.
4. This distance is the continuous Path A score.

```text
Path A score = mean absolute difference(predicted target features, actual target features)
```

#### What is compared with the ground truth?

Path A does not directly predict the binary state `risk` or `not risk`. We first calculate the prediction-error score for every window, then compare the score with the official interval label:

- inside the official risk interval: positive;
- outside the official risk interval: negative.

We then ask whether risk-labeled windows tend to have higher scores, using ROC-AUC and Average Precision. Therefore Path A has no ordinary classification accuracy; it is a ranking test between a continuous prediction-error score and the official risk labels.

The score becomes available after the final 16 frames have been observed. It is not a model that predicts a future accident from the first 48 frames alone, and it does not directly measure feature differences between adjacent windows.

#### Result

ROC-AUC was `0.633691` and Average Precision was `0.037674`, above the constant-score AP of `0.024011`. This supports a limited correlation between prediction error and the risk labels under this setup. Path B was stronger, and Path A produced more false alarms at its selected threshold.

### Path B: frozen features plus a linear probe

#### Hypothesis

Even without using JEPA's prediction function, the features extracted from an observed video window may contain enough information to classify risk and non-risk intervals.

#### Method

1. Feed all 64 observed frames into frozen V-JEPA 2.
2. Average the output over the temporal and spatial dimensions to obtain a 1,024-dimensional global feature.
3. Train a `StandardScaler` and a linear logistic-regression probe using the Nexar training labels.
4. Produce a continuous risk score for every validation window.

Only the probe is trained. V-JEPA 2 remains frozen. Path B uses V-JEPA 2 as a feature extractor, not as a next-feature predictor.

#### What is compared with the ground truth?

The probe score is compared with the same official risk interval labels used by Path A. Unlike Path A, Path B is supervised: the labels are used to fit the probe. Evaluation still focuses on ranking metrics rather than a single thresholded accuracy value.

Path B does not directly detect a large temporal feature change. It classifies each observed window from its feature representation.

#### Result

ROC-AUC was `0.801785` and Average Precision was `0.144671`, substantially above the constant-score AP of `0.024011` and higher than Path A under the same frame evaluation.

## 2. Data and labels

We used 1,500 videos from the official Nexar `train` split: 750 positive and 750 negative videos. The videos were split at the video level into 1,200 internal training videos and 300 internal validation videos. The official test splits were not obtained or used.

For positive videos, the official `time_of_alert` and `time_of_event` values define the half-open risk interval `[time_of_alert, time_of_event)`. A window is positive when its available-at time `t` lies inside that interval. Negative videos contain only negative windows under this labeling rule.

| Data | Videos | Valid windows | Positive windows | Negative windows | Use |
|---|---:|---:|---:|---:|---|
| Internal training | 1,200 | 86,365 | 1,898 | 84,467 | Fit B's scaler and probe |
| Internal validation | 300 | 21,554 | 519 | 21,035 | Select C/thresholds and evaluate A/B |
| Official test | 0 | 0 | - | - | Not obtained |

Windows end at `t`, start at `t - 2.0 s`, and advance by `0.5 s`. Each window score is assigned causally to frames satisfying `t <= u < t + 0.5`. There are 340,025 frame rows in total and 321,931 valid rows, for coverage of `0.946786`.

## 3. Evaluation results

### Frame-level ranking

The common frame set contains 321,931 frames: 7,730 positive and 314,201 negative frames. The constant-score AP, equal to the positive prevalence on the scored set, is `0.024011`.

| Metric | A | B |
|---|---:|---:|
| Common evaluated frames | 321,931 | 321,931 |
| ROC-AUC | 0.633691 | 0.801785 |
| Average Precision | 0.037674 | 0.144671 |
| Video-bootstrap 95% AP interval | [0.030654, 0.046513] | [0.112067, 0.189306] |
| Bootstrap repetitions / seed | 1,000 / 42 | 1,000 / 42 |

The valid A and B frame sets are identical in this run, so the single-path and common-set ranking values are equal. This checks the common evaluation contract; it is not a statistical significance test between the paths.

### Thresholded event metrics

Each path selected its F1-maximizing threshold on the same validation split. These event metrics are therefore exploratory and are not holdout results.

| Metric | A | B |
|---|---:|---:|
| Selected threshold | 0.583661 | 0.899668 |
| F1 at selection | 0.079653 | 0.226540 |
| Events evaluated | 154 | 154 |
| Events detected | 82 | 80 |
| Detection rate | 0.532468 | 0.519481 |
| Mean detection delay | 0.721 s | 1.051 s |
| False alarms | 985 | 161 |
| False alarms / valid second | 0.094263 | 0.015407 |

Although A's detection rate is slightly higher at its selected threshold, it produces substantially more false alarms. The thresholds, alarm grouping, and validation reuse mean these numbers should not be treated as a production-system comparison.

## 4. Interpretation

- Under the current setup, B is stronger for ranking observed two-second windows by their association with the risk labels.
- A is above the constant baseline, so its frozen predictor error may contain information correlated with the risk labels. This does not mean that the predictor error is itself a risk probability.
- Neither path uses future scores to label earlier frames. In Path A, the target frames have been observed by the time the prediction error is computed.
- The experiment does not establish that a large feature change or a pre-accident warning signal was detected. Feature differences, rates of change, and true pre-event prediction require separate designs.

## 5. Limitations and next evaluation

The official test data were not obtained. The B probe hyperparameter and both thresholds were selected on the same validation split used for evaluation. These results are therefore not final unseen-data performance and do not establish generalization, safety, or operational warning quality.

For a final evaluation, keep the B probe, selected C, and thresholds fixed and apply them once to an unused holdout or the official test split. A true early-warning experiment must define labels, available-at times, and metrics that only allow input before the event.

## Reproduction artifacts

- Combined metrics: `artifacts/evaluation-p6-full-ab/metrics.json`
- Combined report: `artifacts/report-p6-full-ab/report.md` / `report.json`
- A scores: `artifacts/prediction-p6-validation/scores.json`
- B scores: `artifacts/scores-p6-full/scores.json`
- Prepared manifest: `artifacts/prepare-nexar-v2/`

See the English README for the official `evaluate` and `report` commands. Raw videos, frames, feature caches, probe weights, and per-video scores are not part of the public release. See [`PUBLICATION_POLICY.en.md`](PUBLICATION_POLICY.en.md) for the release policy.
