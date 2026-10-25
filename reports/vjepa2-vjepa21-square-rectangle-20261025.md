# V-JEPA 2とV-JEPA 2.1のリスク判定・入力形状比較

[English version](vjepa2-vjepa21-square-rectangle-20261025.en.md)

## 要約

同じ固定データ分割と凍結encoderを使い、V-JEPA 2とV-JEPA 2.1を、正方形入力（上下に黒帯あり）と横長入力（黒帯なし）の4条件で比較しました。各条件で学習用データのみからprobeを学習し、同じ40本の検証動画で評価しています。

検証window APは、V-JEPA 2が正方形0.144516、横長0.103784、V-JEPA 2.1が正方形0.057622、横長0.067239でした。正方形から横長へのAP差はV-JEPA 2で−0.040731、V-JEPA 2.1で+0.009617です。動画単位paired bootstrapによる両差の95%区間はいずれも0を含みます。この比較では、横長入力による一貫した性能改善は確認できませんでした。区間が0を含むことは差がない、または同等だという証明ではありません。

V-JEPA 2の横長条件では空間token gridと位置座標が変わります。V-JEPA 2.1ではtoken数を保つ一方、映像部分の解像度を上げています。したがって、観測差を黒帯の有無だけに帰属させることはできず、黒帯が無関係とも結論できません。

## 対象とデータ

これは、Nexar Collision Predictionの公式`train` splitから作成した内部データ分割での、走行動画のリスク区間分類です。公式`test-public`／`test-private`は使用していません。合計200本を動画単位で学習160本と検証40本に分けました。

| 分割 | 動画数 | window数 | 陽性window | 陽性windowを含む動画 |
|---|---:|---:|---:|---:|
| 学習 | 160 | 11,390 | 254 | 81 |
| 検証 | 40 | 2,905 | 65 | 19 |

元動画はすべて1,280×720です。windowは2秒・64フレームで、開始位置を0.5秒ずつずらしました。陽性ラベルは、windowが危険区間に重なるだけではなく、windowの`available_at`時点が危険区間内にある場合に付けています。そのため、検証の危険注釈21動画のうち2本は短いイベントが判定時点にかからず、陽性windowがありませんでした。隣接windowは時間的に75%重なるため、window数を独立した標本数とはみなしません。

この40本は反復利用された固定validation pilotで、未使用の最終holdoutではありません。各条件でvalidationの寸法・probe設定・閾値を選び直していませんが、再利用による選択上の影響は排除できません。

## モデルと4条件

V-JEPA 2はViT-L/16、約300Mパラメータです。モデルIDは`facebook/vjepa2-vitl-fpc64-256`、revisionは[`b3c1679b7c34d3255ef3547f27c7b226aefab26f`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f)です。比較記録では、使用したencoder枝の388 tensorが既存のHugging Face重みと一致し、Hub既定の`target_encoder`枝そのものではないとされています。正方形baselineのAPも既存値と一致しましたが、これだけで異なる実装経路の全出力が数値的に同等とは証明されません。

V-JEPA 2.1はViT-L/16、約300Mパラメータで、約2BパラメータのViT-G教師から蒸留されたEMA studentです。使用checkpointは`vjepa2_1_vitl_dist_vitG_384.pt`、Meta公式コードは[revision `204698b45b3712590f06245fbfba32d3be539812`](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812)です。

| 条件 | モデル入力 | 映像部分 | spatial grid | 時空間token数 |
|---|---:|---:|---:|---:|
| V-JEPA 2・正方形 | 256×256 | 256×144、上下各56pxをpadding | 16×16 | 8,192 |
| V-JEPA 2・横長 | 256×144 | 256×144、paddingなし | 9×16 | 4,608 |
| V-JEPA 2.1・正方形 | 384×384 | 384×216、上下各84pxをpadding | 24×24 | 18,432 |
| V-JEPA 2.1・横長 | 512×288 | 512×288、paddingなし | 18×32 | 18,432 |

## 動画の前処理

全条件で同じ動画、window、フレームIDを使い、PIL BILINEARでRGB映像を処理しました。縦横比を保ったままresizeし、cropや引き伸ばしはしていません。正方形条件は正規化前に黒画素で上下をpaddingしました。黒帯は正方形入力面積の43.75%です。横長条件はpaddingなしです。

V-JEPA 2では、正方形と横長の映像部分はどちらも256×144ですが、token gridは16×16から9×16に変わります。V-JEPA 2.1では、横長条件の映像部分が正方形条件の各辺の4/3、面積の16/9に拡大されます。総token数は同じですが、gridと実画像解像度が異なります。どちらもpaddingだけを独立に変えた試験ではありません。

全条件でRGB値を0〜1にrescaleし、ImageNetの平均値・標準偏差で正規化しました。processor内部の追加resizeとcenter cropは無効です。この比較用letterboxはモデルの公式必須手順ではなく、公式評価の前処理とも異なります。

## Probeと評価方法

encoderは凍結し、最終正規化tokenを時間・高さ・幅方向で平均して特徴量にしました。各条件で同じ手順のprobeを個別に学習しています。

- `StandardScaler`とclass-balanced `LogisticRegression`を学習用データだけでfit
- `C=0.01`、`lbfgs`、seed 42、最大2,000 iteration
- 主指標はwindow AP（Average Precision）
- AP差は検証動画単位のpaired bootstrap、1,000反復、seed 42、percentile 95%区間

APは陽性を高いスコアに順位づけられるかを測る指標で、正解率ではありません。window APとframe APは評価単位が異なるため、数値を直接比較しません。差と区間は丸め前の値から計算し、表示値は小数点以下6桁に丸めています。

## 結果

| 条件 | 学習window AP / ROC AUC | 検証window AP / ROC AUC | 検証frame AP / ROC AUC |
|---|---:|---:|---:|
| V-JEPA 2・正方形 | 0.629291 / 0.991417 | 0.144516 / 0.794117 | 0.104294 / 0.786743 |
| V-JEPA 2・横長 | 0.626648 / 0.992211 | 0.103784 / 0.776376 | 0.073955 / 0.768049 |
| V-JEPA 2.1・正方形 | 0.535919 / 0.986073 | 0.057622 / 0.732140 | 0.050938 / 0.719361 |
| V-JEPA 2.1・横長 | 0.539652 / 0.987757 | 0.067239 / 0.707855 | 0.059994 / 0.701584 |

window AP差は「右側の条件 − 左側の条件」です。

| 比較 | AP差 | 95% percentile区間 |
|---|---:|---:|
| V-JEPA 2・横長 − 正方形 | −0.040731 | [−0.171811, 0.008179] |
| V-JEPA 2.1・横長 − 正方形 | +0.009617 | [−0.081012, 0.063030] |
| 正方形：V-JEPA 2.1 − V-JEPA 2 | −0.086894 | [−0.253621, 0.110627] |
| 横長：V-JEPA 2.1 − V-JEPA 2 | −0.036546 | [−0.127183, 0.070667] |

検証windowの採点coverageは4条件とも2,905/2,905（100%）でした。frameは43,446/45,859（94.738%）が有効で、陽性frameは968件です。評価失敗とprobe収束警告はありませんでした。

## 解釈と限界

- この固定validationでのwindow AP点推定は、横長入力にするとV-JEPA 2で下がり、V-JEPA 2.1で上がりました。しかし両方の95%区間は0を含み、方向や大きさは確定できません。
- frame APはモデルごとにwindow APと同じ方向へ動きましたが、frame ROC AUCは両モデルで低下しました。指標の変化は一様ではありません。
- 正方形条件・横長条件のモデル間差も95%区間が0を含みます。モデルの優劣や同等性を確定できる結果ではありません。
- V-JEPA 2の横長条件ではtoken gridと位置座標が変わり、V-JEPA 2.1ではactive imageの解像度が上がります。padding単独の因果効果は推定できません。
- モデル間では事前学習データ・レシピ、蒸留、解像度、重み、実装経路が異なります。下流probeの学習・検証データを共通にしても、各要因の効果は分離できません。
- これは探索的なリスクwindow分類です。モデル一般の順位、衝突の事前予知能力、実走行環境での安全性を示しません。

## 再現性と公開範囲

本レポートは、公開可能な集計値、条件、モデル識別情報、評価手順を記録するものです。動画、フレーム、動画別スコア、特徴量cache、probe、モデル重みは含みません。レポート単体は再実行パッケージではありません。正確な再現には、Nexarデータの適法な取得、記載したrevision・checkpointに対応する公式モデル、前処理・encoder・probe・集計実装が必要です。外部test split、未使用の独立holdout、推論速度、メモリ、安全性能は評価していません。

## データの出所、引用、利用条件

使用したのは[Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)の`train`由来データです。Nexarの指定引用とデータカードが示す論文を併記します。

> Moura, Daniel C., and Zvitia, Orly. “Nexar Collison Dataset.” Hugging Face, 2025. <https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction>
>
> Daniel C. Moura, Shizhan Zhu, and Orly Zvitia. “Nexar Dashcam Collision Prediction Dataset and Challenge.” 2025. <https://arxiv.org/abs/2503.03848>

データカードはNexar Open Data Licenseを適用し、指定の帰属表示・引用を求めています。営利目的でデータを販売・サブライセンス・再配布するにはNexarの事前書面許可が必要です。悪意あるシステム、危険運転、ディープフェイクや誤情報、再識別・プライバシー侵害、武器化、事故多発地域や個人を利用した搾取的な用途、違法な用途も禁止されています。現在のデータカードでは、データにアクセスする場合は連絡先情報の共有に同意する必要があると説明されています。同意しない場合は取得しないでください。詳細は[確認済みのライセンスrevision](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/ed1ffff7c2dfb9014ff476f079ba5acefc71c72c/LICENSE)を参照し、データ取得・公開時には必ず最新の[公式ライセンス](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE)と[データカード](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/README.md)を確認してください。データの利用条件は、モデルやコードのライセンスとは別です。

V-JEPA 2の仕様は[Meta公式repository](https://github.com/facebookresearch/vjepa2)、[V-JEPA 2.1論文](https://arxiv.org/html/2603.14482v1)、[Transformers V-JEPA 2 documentation](https://huggingface.co/docs/transformers/model_doc/vjepa2)も参照してください。外部モデルファイルは本レポートに含みません。
