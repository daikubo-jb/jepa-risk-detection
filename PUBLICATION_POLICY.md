# データ・モデル・公開方針

更新日: 2026-09-15

英語版: [`PUBLICATION_POLICY.en.md`](PUBLICATION_POLICY.en.md) / 第三者表示: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

この文書は、`jepa-risk-detection`を公開する際に、実験コード、外部データ、外部モデル、生成物を分けて扱うための方針です。これは法律意見ではなく、公開前に権利者・所属先・利用規約を再確認するための実務上のチェックリストです。

English version: [`PUBLICATION_POLICY.en.md`](PUBLICATION_POLICY.en.md)

## 結論

- Phase 1で実際に使用したデータセットはNexar Collision Predictionの`train`だけです。公式`test-public`／`test-private`は使用していません。
- MP4、Parquet、抽出フレーム、特徴量cache、動画別スコア、preview画像は、初回の公開リポジトリに含めません。Bのprobeモデルは公開候補としますが、Nexarデータから学習した派生モデルとして個別確認を通過した場合だけ含めます。
- モデルはモデルID、固定revision、取得手順、出所を記録します。大容量のモデルファイルは同梱しません。
- A経路で使うMeta公式のraw predictor checkpoint `vitl.pt`は、個別の適用ライセンスを確認できるまで再配布・ミラー・リポジトリへの同梱を行いません。
- 集計結果と説明文は、個別動画を特定できる情報、フレーム画像、絶対パス、秘密情報を含まないことを確認したうえで公開します。
- プロジェクト独自コード用のルート`LICENSE`としてMIT Licenseを追加し、著作権表示を`daikubo-jb`としました。このライセンスは独自コードだけに適用されます。

## 1. Nexar Collision Prediction

### 出所

- [公式データセットページ](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)
- [公式データカード](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/README.md)
- [Nexar Open Data License](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE)

この実験では、公式`train`の1,500本（positive 750本、negative 750本）を使用します。positiveの`time_of_alert`と`time_of_event`から、リスク区間`[time_of_alert, time_of_event)`を作ります。データセットの目的、ラベル、引用方法は公式データカードを正とします。

### 利用・表示上の条件

2026-09-15に確認した公式ライセンスrevisionは`ed1ffff7c2dfb9014ff476f079ba5acefc71c72c`です。公式ライセンスに従い、少なくとも次を守ります。

- Nexarの著作権表示と指定された引用情報を保持・掲載する。
- データセットを販売、再許諾、または営利目的で再配布しない。営利目的の再配布が必要な場合は、事前にNexarの書面による許可を得る。
- 悪意のあるシステムや危険運転、ディープフェイク・誤情報、再識別・プライバシー侵害、武器化などの用途に使わない。
- 事故多発地域や個人を利用した搾取的な保険・金融その他の目的に使わない。
- 適用される法令を守る。
- データは提供時の状態で利用し、公式ライセンスの免責事項を前提とする。

このリポジトリはNexarデータの利用権を再許諾しません。データを取得する利用者は、取得時点の公式ライセンスを自分で確認してください。公開リポジトリには生データを含めず、公式取得先とライセンスへのリンクだけを掲載します。

### 公開しないもの

次のファイルは、ライセンス・プライバシー・再配布範囲を個別確認しない限り公開しません。

- NexarのMP4、Parquet、抽出フレーム
- 元アノテーションを含む詳細manifest
- 動画ID単位のスコア、失敗例、サムネイル、確認画像
- 元データから直接計算した特徴量cacheや、それを含む圧縮アーカイブ

集計した評価値や再現手順を公開する場合も、個別動画を推測できる情報、ローカル絶対パス、取得用Cookieを含まないことを確認します。

## 2. V-JEPA 2

### HF encoder

経路Bでは次のモデルを使用します。

- Model: `facebook/vjepa2-vitl-fpc64-256`
- Revision: `b3c1679b7c34d3255ef3547f27c7b226aefab26f`
- [固定revisionの公式モデルページ](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f)

指定revisionのモデルカードにはMITライセンスが表示されています。公開文書では、モデルIDとrevisionを固定して記載し、利用者が公式配布元から取得できるようにします。モデルカードの条件が変更された場合は、公開前に再確認します。

### Meta公式コードと独自実装

Metaの[V-JEPA 2公式リポジトリ](https://github.com/facebookresearch/vjepa2)は、大部分がMITライセンスです。一方、公式READMEは次のファイルをApache-2.0として扱っています。

- `src/datasets/utils/video/randaugment.py`
- `src/datasets/utils/video/randerase.py`
- `src/datasets/utils/worker_init_fn.py`

このプロジェクトの`src/jepa_risk/predictor.py`は、Metaのソースファイルを同梱したものではなく、公式のアーキテクチャとcheckpointのstate dict名に合わせた独自の互換実装として分類します。現行ツリーには、公式READMEがApache-2.0として指定する3つのユーティリティファイルの転載はありません。Metaの実装への帰属と固定commitは第三者表示に残します。将来Metaのソースを転載・翻案する場合は、対象ファイルの著作権表示とMITまたはApache-2.0の条件を保持します。公式実装のライセンスが、データセットや別配布の重みの権利を自動的に与えるとは解釈しません。

### A経路のraw predictor checkpoint

A経路では、Meta公式READMEから案内されている次のcheckpointを使用します。

- Source: `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt`
- Project-pinned source commit: `204698b45b3712590f06245fbfba32d3be539812`
- Current local SHA-256: `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012`
- Current local size: `5,127,726,842` bytes（約4.78 GiB / 5.13 GB）

上記のraw `.pt`ファイルについて、今回確認した公式ページだけでは個別の適用ライセンスを確定できませんでした。そのため、公開リポジトリには含めず、利用者が公式配布元から自分で取得する方式にします。サイズとSHA-256は、公開直前に再計算して更新します。raw checkpointを商用利用、再配布、派生重みの公開に利用できるとは、この文書から判断しません。

## 3. 生成物と再現性

公開候補は次の優先順で扱います。

1. ソースコード、設定雛形、CPUで実行できる合成テスト、実験の説明
2. 外部データ・外部モデルの公式取得手順、revision、commit、SHA-256
3. 個別動画を含まない集計レポートと、公開可能性を確認した評価結果

初回公開から除外するものは次のとおりです。

- 外部データとその抜粋
- 外部モデルcheckpoint
- 特徴量cache、動画別の詳細スコア、Aのraw predictor checkpoint
- フレーム画像、動画プレビュー、Cookie、token、ローカル環境の絶対パス

`.gitignore`だけではGitHubのrelease添付、LFS、wheel、sdist、圧縮アーカイブへの混入を防げません。公開前に、Git管理対象、履歴、release添付、配布物を個別に検査します。

## 4. プロジェクト独自コード

ルート`LICENSE`には独自コード向けのMIT Licenseを追加し、著作権表示を`daikubo-jb`としています。公開前に、`daikubo-jb`が独自コードをMITで許諾できることを確認します。共同著作者・所属先など別の権利者が関係する場合、または権利関係が不明な場合だけ追加確認を行います。独自コード用ライセンスは、Nexarデータ、V-JEPA 2モデル、raw checkpoint、Python依存パッケージの条件を上書きしません。

Bのprobeモデルを公開する場合は、元データを含まない係数・scaler統計・設定だけを安全な形式で配布し、学習データ、特徴量cache、動画別スコアを同梱しません。Nexarライセンスが学習済みprobeの再配布を明示していないため、公開前にこの派生モデルの公開可否を別途確認します。`pickle`のままではなく、JSONまたはNPZなどの形式を優先します。

## 5. 公開前チェックリスト

- [x] 独自コード用のルート`LICENSE`と第三者表示文書を追加した
- [ ] Nexarの公式ライセンス、データカード、指定引用を公開文書から参照できる
- [ ] MP4、Parquet、フレーム、特徴量cache、動画別出力、raw checkpoint、Cookieが公開対象から除外されている
- [ ] B probeモデルの派生・再配布可否を確認し、安全な配布形式とモデルカードを用意した
- [ ] `vitl.pt`の個別条件を再確認し、再配布不可の暫定方針を更新した
- [ ] Meta由来コードの転載・翻案範囲を確認し、該当するMIT/Apache-2.0表示を追加した
- [ ] Git履歴、LFS、release添付、wheel、sdist、圧縮アーカイブを検査した
- [ ] READMEとレポートに、公式test未評価・実環境の安全性能未検証であることを明記した
- [ ] 公開する集計結果に絶対パス、個別動画の識別情報、画像、秘密情報がない

## English summary

This project uses only the `train` split of the Nexar Collision Prediction dataset for Phase 1. The official public and private test splits are not used. Raw videos, Parquet metadata, frames, feature caches, per-video scores, preview images, cookies, and raw model checkpoint files are excluded from the initial public repository. The B probe is a possible release artifact, subject to a separate derived-model rights check.

The Nexar dataset is governed by the [Nexar Open Data License](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE). Keep the required copyright and citation information. Do not sell, sublicense, or redistribute the dataset for profit without prior written consent from Nexar. Do not use it for malicious systems or unsafe driving, deepfakes or misinformation, re-identification or privacy violations, weaponization, or other unlawful purposes. Users must review the current official license themselves.

The encoder is `facebook/vjepa2-vitl-fpc64-256` at revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`; its model card shows an MIT license. The majority of the official V-JEPA 2 code is MIT, while the three video utility files listed above are Apache-2.0. Any copied or adapted Meta code must retain the applicable notices.

The A-path raw predictor checkpoint `vitl.pt` is obtained from the official Meta URL and identified by the pinned source commit and SHA-256 above. Its separate terms were not conclusively identified in the reviewed official pages, so this repository does not redistribute or mirror it. The hash and size must be rechecked immediately before publication.

The project's own code is released under the root [MIT License](LICENSE), with `daikubo-jb` as the copyright holder. Before publication, confirm that `daikubo-jb` is authorized to grant this license for the project-owned code. Additional review is needed only if co-authors, an employer, or another rights holder is involved or the ownership is uncertain. The project license does not override the terms of the dataset, model, checkpoint, or dependencies. This policy is a publication gate, not legal advice.
