# jepa-risk-detection

English version: [`README.en.md`](README.en.md)

V-JEPA 2 の凍結特徴量と予測誤差を使い、Nexar Collision Predictionの走行映像で衝突・ニアミスのリスク予測を検証する実験プロジェクトです。

Pythonの実行環境は `uv` で管理します。

```sh
uv python pin 3.12
uv lock
uv sync --locked
uv run pytest -m "not gpu"
uv run python -m jepa_risk.cli preflight \
  --config configs/phase1.yaml \
  --output artifacts/preflight

uv run python -m jepa_risk.cli prepare \
  --config configs/phase1.yaml \
  --output artifacts/prepare

# まず1動画・1窓で実モデルとcacheを確認する
uv run python -m jepa_risk.cli extract \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare \
  --output artifacts/features-smoke \
  --split train \
  --selection smoke20 \
  --limit-videos 1 \
  --max-windows 1 \
  --local-files-only
```

`preflight.json`には、uv・Python・PyTorch・GPUバックエンド、データセットのパス、`ffprobe`・`pyarrow`の利用可否、モデル設定を記録します。V-JEPA 2のpredictor checkpointが未配置でも、配置状況を`checks`に残します。

`prepare`は設定されたデータセットのメタデータからmanifest・split・窓定義を生成します。Nexarでは、Parquetの`time_of_alert`から`[time_of_alert, time_of_event)`をリスク区間として扱い、MP4ごとの実fpsとフレーム数を`ffprobe`で取得します。

`extract`はPyAVで窓のRGBフレームを読み、設計書のletterbox（256×256、BILINEAR、黒余白）後に公式`VJEPA2VideoProcessor`で正規化し、`get_vision_features()`の`[B,8192,1024]`を全体平均`[B,1024]`と時間平均`[B,32,1024]`へ集約します。窓内のフレームは半開区間`[observation_start_s, available_at_s)`だけから選びます。`--selection smoke20`はmanifestに保存したseed=42の固定20本、`pilot`は固定予備集合を使います。cacheはモデルrevision・manifest・窓定義・前処理・pooling設定のSHA-256キー単位で保存し、未完成ディレクトリは再利用しません。`--limit-videos`／`--max-windows`を外す前に、実行時間と保存容量を確認してください。

train/validationのcacheが揃った後は、trainのみでscalerと分類器をfitし、validationのframe APでCを選びます。validationで決めた閾値はtestへ固定適用します。

```sh
uv run python -m jepa_risk.cli train-probe \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare-nexar-v2 \
  --features artifacts/features-v3-p4 \
  --output artifacts/probe-p4-smoke \
  --train-selection smoke20 \
  --validation-selection pilot \
  --train-limit-videos 20 \
  --validation-limit-videos 4

uv run python -m jepa_risk.cli score-probe \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare-nexar-v2 \
  --features artifacts/features-v3-p4 \
  --model artifacts/probe-p4-smoke/probe.pkl \
  --output artifacts/scores-p4-smoke \
  --split validation \
  --selection pilot \
  --limit-videos 4
```

`train-probe`の`--limit-videos`は両splitに同じ上限を適用する互換オプションです。splitごとに異なる上限を使う場合は`--train-limit-videos`と`--validation-limit-videos`を指定します。

ここで生成される`probe.pkl`はローカル実験用のpickle成果物であり、初回公開リポジトリには含めません。B分類モデルを公開する場合は、権利確認後にJSON/NPZ等の安全な形式へ書き出し、学習データ・特徴量cache・動画別スコアを含めないモデルカード付き成果物として別途用意します。

Bの全件validation結果の解釈は[`b-result.md`](b-result.md)にまとめています。A/B全件validationの比較結果は[`ab-result.md`](ab-result.md)にまとめています。Bは特徴量の時間差分を直接検出するのではなく、観測済み窓のglobal特徴からリスクラベルを線形に識別できるかを検証する経路です。

英語版レポート: [`ab-result.en.md`](ab-result.en.md)

経路A（Meta公式predictorのマスク予測誤差）は、`model.checkpoint`に公式`vitl.pt`を指定して実行します。checkpointが未配置の環境では、実行前に取得してください。`score-prediction`は先頭24時刻tokenをcontext、残り8時刻tokenをtargetとし、target encoderのLayerNorm後特徴との平均L1を窓末尾時刻へ出力します。

```sh
uv run python -m jepa_risk.cli score-prediction \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare-nexar-v2 \
  --output artifacts/prediction-p5-pilot-v2 \
  --split validation \
  --selection pilot \
  --limit-videos 4 \
  --local-files-only
```

長時間の実行では`--resume`を付けると、成功済み動画のcheckpoint（`output/checkpoints/<cache_key>/<video_id>.json`）を再利用できます。初回から`--resume`を付けて開始し、動画単位で再開できるようにします。

validation全件を実行する場合は、次のコマンドを使います。

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

公式ViT-L checkpointは約5.13GB（5,127,726,842 bytes、現在のローカル取得値）です。公開前にサイズとSHA-256を再確認してください。予測誤差スコアは確率ではなく、validationで閾値を選ぶ場合はラベル調整を行った値として扱います。

実測時間・全件窓数・特徴cache容量・A/B評価・失敗区間を統合するには、`report`を使います。これは全件抽出を開始せず、指定したmeasurementから線形見積もりを作ります。

```sh
uv run python -m jepa_risk.cli report \
  --config configs/phase1.yaml \
  --prepared artifacts/prepare-nexar-v2 \
  --output artifacts/report-p6-pilot \
  --feature-measurement artifacts/features-20-full/extraction-measurement.json \
  --prediction-measurement artifacts/prediction-p5-20-smoke-v2/scores.json \
  --probe-metrics artifacts/evaluation-p4-smoke/metrics.json \
  --prediction-metrics artifacts/evaluation-p5-pilot-v2/metrics.json \
  --probe-scores artifacts/scores-p4-smoke/scores.json \
  --prediction-scores artifacts/prediction-p5-pilot-v2/scores.json
```

`report.json`と`report.md`には、現在の全107,919窓に対する処理時間・容量見積もり、共通フレーム評価、閾値イベントの誤報・見逃し・部分採点例、公式test未取得などの制約が保存されます。

保存済みのA/Bスコアから評価メトリクスを再生成する場合は、次のコマンドを使います。A/Bの共通フレーム集合、動画単位bootstrap、経路別の閾値イベント指標を`metrics.json`へ保存します。

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

全件validationの統合レポートは`artifacts/report-p6-full-ab/`に生成済みです。公式test未取得のため、現時点の指標は探索的なvalidation結果です。

## Nexar trainデータ

Nexarのtrainデータは公式Hugging Face配布から取得します。公式test-public / test-privateは取得せず、学習用1,500本だけを使います。取得済みデータの保存先は`artifacts/data/nexar_train`です。

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

Nexarの動画とParquetは、[公式データカード](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/README.md)および[公式LICENSE](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE)に従って利用します。動画はGit管理対象外です。

## 公開方針

独自コードは[`LICENSE`](LICENSE)のMIT Licenseで公開します。データ、モデル、生成物の利用条件と公開対象は[`PUBLICATION_POLICY.md`](PUBLICATION_POLICY.md)にまとめています。英語版は[`PUBLICATION_POLICY.en.md`](PUBLICATION_POLICY.en.md)です。第三者表示は[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)にまとめています。生データ、フレーム、特徴量cache、モデルcheckpoint、Cookieはリポジトリへ含めません。
