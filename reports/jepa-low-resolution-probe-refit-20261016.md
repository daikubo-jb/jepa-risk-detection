# 低解像度条件でprobeを再学習したV-JEPA 2評価

[English version](jepa-low-resolution-probe-refit-20261016.en.md)

GitHub版: [日本語レポート](https://github.com/daikubo-jb/jepa-risk-detection/blob/main/reports/jepa-low-resolution-probe-refit-20261016.md) / [英語レポート](https://github.com/daikubo-jb/jepa-risk-detection/blob/main/reports/jepa-low-resolution-probe-refit-20261016.en.md)

**対象実験:** Nexar Collision Predictionの内部train/validationから選んだ160本/40本の固定pilot

**集計:** 2026-10-14の3条件比較

## 要約

V-JEPA 2のencoder（映像を数値表現である特徴量に変換するモデル）を凍結、つまり学習済みの重みを更新しないまま、入力映像の空間的な細部を256、128、64の3条件で変え、条件ごとに同じ設定の線形probe（特徴量からリスクのスコアを出す小さな判定器）だけを再学習しました。validation（学習に使わない検証用データ）の窓単位（window：動画から切り出した時間区間）Average Precision（AP）は256条件で0.144516、128条件で0.107617、64条件で0.101968でした。128と64の点推定は256より低い一方、動画単位paired bootstrap（同じ動画の再標本化を条件間で揃える方法）による差の95%信頼区間（標本の変動に伴う不確かさを表す区間）はいずれも0を含みます。今回のpilotだけでは、差の向きや大きさを確定できません。

APは、リスク区間という正例をスコア上位に並べられるかを、precision（適合率：正例と判定した中で実際に正例だった割合）とrecall（再現率：実際の正例のうち拾えた割合）から評価します。正解率とは異なり、AP=0.144516を「約14%の判定が正しい」とは読みません。

これは入力の細部を減らす実験です。128/64へ縮小した画像を256へ再拡大してencoderへ入力するため、encoderの入力サイズとtoken gridは全条件で同じです。推論時間やメモリ使用量の削減を測る実験ではありません。

## 用語と評価指標

| 用語 | このレポートでの意味 |
|---|---|
| encoder・特徴量・凍結 | encoderは映像を数値表現である特徴量に変換するモデルです。凍結とは学習済みの重みを更新しないことで、この実験では全条件で同じV-JEPA 2を使います |
| 線形probe | 特徴量に重みづけしてリスクのスコアを出す小さな判定器です。ここではStandardScalerで特徴量を標準化し、LogisticRegressionを学習します。どちらもtrain split（学習用データ）のみでfitします。条件ごとに学習し直すのはこのprobeの処理系で、V-JEPA 2自体ではありません |
| precision / recall | precision（適合率）は正例と判定した中の本当の正例の割合、recall（再現率）は実際の正例のうち拾えた割合です。ここでの正例はリスク区間ラベルのある評価対象を指します |
| AP / PR-AUC | AP（Average Precision）は閾値を動かしたときのprecisionを、recallの増分で重みづけして足し合わせるランキング指標です。PR-AUC（precision-recall曲線の下面積）を台形積分で計算すると点間を直線補間するため、APとは算出法が異なり値が一致するとは限りません。本表はAPを報告しています |
| 陽性率・評価集合 | 陽性率は評価対象に正例が占める割合です。APは陽性率や評価集合の構成に依存します。同じ評価集合・ラベル・単位を揃えて比較し、別データのAPをそのまま比較しません |
| window / frame単位評価 | windowは切り出した時間区間、frameは動画を構成する1枚の画像です。窓APは窓ごと、フレームAPは有効なフレーム行ごとのスコアとラベルから計算します。単位や陽性率が異なるため、両者の数値の大小を直接性能差と読みません |
| paired bootstrap・95%信頼区間 | validation動画を動画単位で復元抽出（選んだ動画を候補に戻し、重複を許して選ぶこと）し、各回で同じ動画の組を条件間で共有してAP差を計算します。1,000回の差の分布の下側2.5%点と上側97.5%点を95% percentile区間とします。再標本化は窓を独立に選び直すものではありません。これは標本の変動に伴う不確かさの推定で、真の差がこの区間に入る確率を95%と断定する意味でも、0を含めば同等性が証明されるという意味でもありません |
| train / validation・pilot | trainはprobeの学習用、validationは学習に使わない検証用です。pilotは小規模な探索的評価であり、今回は固定した40本を使っています |
| stride・letterbox・token grid | strideは時間窓をずらす間隔です。letterboxは縦横比を保って余白を足し、画像を所定サイズに収める処理です。token gridはencoderが扱う画像の小領域の並びで、今回の縮小・再拡大では入力サイズとともに全条件で固定です |
| 閾値・イベント検出・false alarms | 閾値はスコアから検出の有無を決める境界です。イベント検出数とfalse alarms（誤警報数）は、その閾値での検出をイベントと対応づけて数える指標で、スコアの順位を評価するAPとは別です |

用語の参考資料：AP・precision・recallと台形積分の違いは[scikit-learnの指標解説](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)、陽性率との関係は[precision-recall曲線の仕様](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)、復元抽出とpercentile区間の一般的な説明は[SciPyのbootstrap解説](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)を参照してください。これらは定義の参照先で、今回の実装や実行ライブラリを示すものではありません。

## 実験の問い

この比較では、二つの問いを扱います。

1. 同じ凍結encoderに対して入力画像の空間的な細部を減らすと、リスク区間の窓を順位づけるAPはどう変わるでしょうか。
2. 各detail条件でprobeを再学習した今回の設定では、APはどう変わるでしょうか。

ここで変えるのは入力画像に残る空間情報です。V-JEPA 2自体の重みや時間方向の窓の切り方は変えません。

## 実験条件

| 項目 | 条件 |
|---|---|
| データセット | Nexar Collision Predictionの公式`train` splitから作成した内部split。公式test splitは使用していません |
| 学習側 | 160本、11,390窓（正例254、負例11,136） |
| validation側 | 同じ固定pilot 40本、2,905窓 |
| split重複 | 0本 |
| モデル | `facebook/vjepa2-vitl-fpc64-256`、revision [`b3c1679b7c34d3255ef3547f27c7b226aefab26f`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f) |
| 時間窓 | 2秒、64フレーム、0.5秒stride |
| encoder | 凍結。条件間で同一 |
| probe | StandardScalerとclass-balanced LogisticRegressionの両方をtrain splitのみでfit。C=0.01、lbfgs、seed=42 |
| 入力の処理 | 256×256にletterbox後、256/128/64のdetail条件を作成。128/64はBILINEARで縮小し、256×256へ再拡大してから同じencoderへ入力 |
| 主な評価 | 窓単位AP。差の区間はvalidation動画単位のpaired bootstrap、1,000回、seed=42 |

全条件でvalidationの窓は2,905件採点されました。128/64条件の特徴抽出失敗は0件でした。フレーム指標では各条件とも45,859行のうち43,446行（94.74%）が有効でした。イベント21件はすべて全区間を採点できました。

## 結果

| detail条件 | 窓AP | 256との差（95% bootstrap区間） | フレームAP | 固定閾値でのイベント検出 | False alarms |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.144516 | — | 0.104294 | 4 / 21 | 20 |
| 128 | 0.107617 | -0.036898 `[-0.184652, 0.083747]` | 0.082793 | 4 / 21 | 25 |
| 64 | 0.101968 | -0.042548 `[-0.199979, 0.072646]` | 0.083509 | 2 / 21 | 18 |

イベント列は、3条件で共通の閾値0.8996675を使った参考値です。この閾値の選定履歴は今回確認していないため、運用性能の比較としては扱いません。

AP差の点推定は128、64ともに負でしたが、両方の95%区間が0をまたいでいます。この結果から「低解像度で有意に悪化した」とは結論できません。今回は低解像度条件にも合わせてprobeを学習し直していますが、別の学習規模で作成された過去の固定probe結果とは条件が異なるため、「再学習で精度が回復した」とも言えません。

## 公開状況と再現性

この比較に使った`scripts/resolution_ablation.py`と、`src/jepa_risk/preprocessing.py`の対応する追加変更は、現時点で未公開です。本レポートは保存済みの実験結果の報告であり、現在公開中のリポジトリをcloneするだけで同じ実験を再現できる状態ではありません。

この版の比較スクリプトの標準summaryはフレーム単位の指標と差を集計します。本表の窓APと窓AP差の95%信頼区間を再計算するための窓単位集計手順も、公開準備上の残件です。実行コードと集計手順の公開・検証は別途必要です。

## 再現の要点

- Python 3.12の設定、依存関係、準備手順はプロジェクトの[`README.md`](../README.md)と`uv.lock`を参照してください。
- 比較処理の実装は未公開の`scripts/resolution_ablation.py`です。前処理のファイルは[`src/jepa_risk/preprocessing.py`](../src/jepa_risk/preprocessing.py)ですが、本比較用の細部削減の追加変更は未公開です。
- 実行モードは`--mode pilot --probe-mode refit --selection pilot`、比較条件は256/128/64、bootstrapは1,000回・seed 42です。設定は[`configs/phase1.yaml`](../configs/phase1.yaml)に固定されたモデルID・revisionと、2秒/64フレーム/0.5秒strideを使います。
- Nexarのローカル取得データ、準備済みmanifest、照合対象の256特徴・probe metadata、既存評価の閾値情報、モデル重みなどが実行時に必要です。これらのデータ・重み・特徴cache・動画別スコアはこの文書とともに配布しません。初回公開方針は[`PUBLICATION_POLICY.md`](../PUBLICATION_POLICY.md)を参照してください。
この比較スクリプトはB経路のprobe再学習に加えてA経路も実行するため、既存A経路のbaselineスコアと公式predictor checkpoint、および256条件のprobe本体・metadataも必要です。これらのファイルは文書とともに配布しません。

- 再実行時は未使用の出力先を指定してください。既存出力は上書きしない設計です。

## 解釈上の限界

- 評価対象は40本の固定pilotのみです。公式test splitや独立した大規模holdoutでの評価ではありません。
- AP差のbootstrap区間は0を含み、今回の比較から統計的な差を断定できません。
- 256/128/64ごとにprobeをtrain160で学習しています。過去に1,200本で学習した固定probeの数値とは直接比較できません。再学習による回復の証拠にはなりません。
- イベント指標は選定履歴未確認の同一閾値による参考比較です。閾値の選び直しや実運用上の閾値評価は行っていません。
- これはリスク区間ラベルを窓単位で分類する探索的評価です。事故を事前に予知できること、実際の運転環境で安全に機能することは示しません。
- 入力の細部を減らしてもencoderへのサイズは256×256のままです。推論時間・メモリ使用量・消費電力の削減は評価していません。

## データ、モデル、ライセンス

使用したデータはNexar Collision Predictionです。Nexar指定のdataset citationと論文を参照してください。データセットは本プロジェクトのMIT Licenseの対象ではありません。取得時点の公式ライセンスを利用者自身で確認してください。動画やデータの派生物はこの成果物に含めていません。

- [Nexar Collision Prediction dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)
- [Nexar Open Data License（revision `ed1ffff7c2dfb9014ff476f079ba5acefc71c72c`）](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/ed1ffff7c2dfb9014ff476f079ba5acefc71c72c/LICENSE)
- [Nexar dataset paper, arXiv:2503.03848](https://arxiv.org/abs/2503.03848)
- [V-JEPA 2 model card at the configured revision](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f) — このrevisionのHugging FaceページにはMITと表示されています。

本プロジェクトのコード向けMIT Licenseは、外部データ・モデルの条件やprobeなどの派生物の再配布可否を決めるものではありません。今回のレポートは集計値と再現情報に限り、動画、フレーム、モデル重み、probe、特徴量、動画単位の出力は配布しません。派生probeの再配布条件は未確認です。

## 引用

> Moura, Daniel C., and Zvitia, Orly. “Nexar Collison Dataset.” Hugging Face, 2025. <https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction>

> Daniel C. Moura, Shizhan Zhu, and Orly Zvitia. “Nexar Dashcam Collision Prediction Dataset and Challenge.” 2025. <https://arxiv.org/abs/2503.03848>
