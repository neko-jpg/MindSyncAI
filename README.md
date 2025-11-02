# 音声感情認識モデル開発ガイド

このリポジトリは、音声データから感情を分類するTensorFlow Liteモデルを開発するためのものです。
学習にはRAVDESSおよびCREMA-Dデータセットを使用します。

## 成果物

- `notebooks/training.ipynb`: データの前処理、モデルの学習、評価、TFLite変換までを行うJupyter Notebook。
- `models/`: 学習済みモデル (`.h5`, `.tflite`) とラベルファイル (`labels.txt`) が格納されます。
- `inference.py`: TFLiteモデルを使用して、単一の音声ファイルから感情を推論するスクリプト。
- `requirements.txt`: 必要なPythonライブラリ。

## 1. セットアップ

### a. リポジトリのクローン
```bash
git clone <repository-url>
cd <repository-name>
```

### b. 依存ライブラリのインストール
```bash
pip install -r requirements.txt
```

### c. データセットの準備
1. [RAVDESS](https://zenodo.org/record/1188976) と [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) の音声データセットをダウンロードします。
2. ダウンロードしたデータセットを、プロジェクトルートの `data/` ディレクトリに配置します。
   ディレクトリ構造の例:
   ```
   .
   ├── data/
   │   ├── RAVDESS/
   │   │   ├── Actor_01/
   │   │   │   └── 03-01-01-01-01-01-01.wav
   │   │   └── ...
   │   └── CREMA-D/
   │       ├── 1001_DFA_ANG_XX.wav
   │       └── ...
   ├── models/
   ├── notebooks/
   └── ...
   ```
   **注意:** `training.ipynb` は `data/` ディレクトリ内を再帰的にスキャンするため、サブディレクトリの構造は上記と異なっていても問題ありません。

## 2. モデルの学習と変換

1. Jupyter Notebookを起動します。
   ```bash
   jupyter notebook
   ```
2. ブラウザで `notebooks/training.ipynb` を開きます。
3. 「Cell」メニューから「Run All」を選択し、ノートブック全体を実行します。

実行が完了すると、`models/` ディレクトリに以下のファイルが生成されます。
- `emotion_model.h5`: 学習済みのKerasモデル
- `emotion_model.tflite`: 量子化済みのTensorFlow Liteモデル
- `labels.txt`: 感情ラベルのリスト

## 3. 推論の実行

`inference.py` スクリプトを使用して、単一の音声ファイル (`.wav`) の感情を推論できます。

```bash
python inference.py path/to/your/audio.wav
```

**実行例:**
```
$ python inference.py data/RAVDESS/Actor_01/03-01-03-01-01-01-01.wav

'data/RAVDESS/Actor_01/03-01-03-01-01-01-01.wav' の感情を推論します...

推論結果: happy (信頼度: 0.98)
------------------------------
各感情の確率:
- angry: 0.001
- calm: 0.003
- disgust: 0.000
- fearful: 0.002
- happy: 0.981
- neutral: 0.005
- sad: 0.001
- surprise: 0.007
```

---
---

# MindSync用 AIモデル要件定義（音声感情推定・オンデバイス推論）
1. 目的・スコープ

目的: 1日1回・最大60秒の音声から、気分スコア（0–1）と感情の色（カテゴリ）を出力し、履歴可視化・マッチング・ミニゲーム推薦に利用する。

README

スコープ: 端末内（オフライン）推論を主とし、必要に応じてサーバ側で再解析・モデル配信を行う。

README

2. 入力要件（Audio I/O）

入力音声: モノラル、推奨16 kHz / 16-bit PCM、長さ3–60秒（最短3秒未満は低信頼扱い）。

前処理（端末/Worker両対応）:

VAD（無音区間除去）、正規化、ノイズ軽減（簡易スペクトル減算）。

特徴量: log-melスペクトログラムまたはMFCC（20–40次元）。

失敗時処理: 無音・過大入力・極端なSNR低下時は「信頼度低」のフラグを返す。

3. 出力要件（API／UI連携）

主要出力

mood_score: float [0,1]

emotion_probs: dict（例: {joy, sadness, anger, fear, neutral, …}）

emotion_color: カラーマップ（例: joy→Yellow、sadness→Blue 等）

confidence: float [0,1]

副出力

quality_flag: {ok, low_energy, noisy, too_short}

version: モデル/特徴量/前処理のセマンティックバージョン

UI反映: /timelineで履歴グラフ、/matchで近傍ユーザー推薦、/gamesで状態適合のゲーム提示に利用。

README

4. モデル要件（学習〜TFLite）

アーキテクチャ（推奨いずれか）

2D-CNN（log-mel入力）または 1D-CNN + GAP（MFCC系列）＋小規模FC

サイズ目標: ≤ 5 MB（理想3MB以下）、int8/float16量子化対応

推論レイテンシ: mid-tier端末で**≤ 200 ms/推論**

RAMフットプリント: ≤ 32 MB（モデル＋入出力テンソル）

移植形式: .tflite（必須）、将来のバックエンド最適化用にONNXも任意。

README

プラットフォーム: Flutter（tflite_flutter）、NNAPI/Core ML Delegate対応

5. 学習データ・ラベリング要件

感情クラス（初期案）: neutral, joy, sadness, anger, fear, surprise（6–8クラスで検討）

データ源: 公開コーパス（RAVDESS/CREMA-D等）＋拡張（日本語話者の追加収録）

言語/話者バランス、性別/年齢レンジ、録音環境（静音/環境音）を分布管理

データ数目安: クラスあたり**≥ 1,000 発話**（拡張は段階的）

拡張: ピッチシフト、タイムストレッチ、環境ノイズ付与（SNR 5–20dB）

6. 評価基準（Acceptance）

一次指標: Macro-F1 ≥ 0.70（バリデーション）、AUROC ≥ 0.80

二次指標: 推論時間、モデルサイズ、端末消費電力（連続10回推論で温度上昇が小さいこと）

ロバスト性: 雑音・小声・方言・笑い声等での性能劣化を定量化（ノイズ条件別F1）

失敗時体験: confidence < τ で「中立表示＋再録音ガイド」を返す

7. 前後処理仕様（厳密化）

前処理: 16 kHzへ再サンプル → VAD → 正規化 → 特徴抽出（[T x F]）

後処理:

softmax出力の温度スケーリングで信頼度較正

mood_scoreは w·emotion_probs の線形写像＋滑らか化（EMA）

emotion_colorはクラス→色の辞書で決定（UIテーマと統一）

8. 端末内推論・統合要件

実装: EmotionAnalyzer（Dart）で初期化・推論・後処理をカプセル化

init()でInterpreter非同期ロード（スプラッシュ中に実行）

analyze(mfcc|mel)が{mood_score, probs, color, confidence, version}返却

状態管理: Riverpodで提供、結果はIsarに永続化し/timelineへ反映

フォールバック: Interpreterロード失敗時は「簡易ヒューリスティック」提示

リソース: モデルはassets/models/emotion_model.tfliteに同梱

9. バックエンド連携・更新

構成: Supabase（Auth/DB/Storage/Realtime）＋Worker(API: FastAPI/Cloud Run)

音声前処理・再解析・マッチング（pgvector）に利用。

README

モデル更新: 起動時にStorageのlatest.jsonをチェック → 差分DL → 検証 → 切替（ロールバック可）

データ同期: emotion_logsはRLS適用の上で暗号化・最小化して送信。

README

10. セキュリティ・プライバシー

原則: 推論は端末内完結。音声生データはユーザー選択時のみ送信

保存: 解析結果のみIsarに保存、必要最小限をSupabaseへ同期（RLS）

README

コンプライアンス: データセットライセンス順守、医療用途の不使用明記（ウェルネス用途）

11. テスト計画

単体: 前処理関数（VAD/MFCC）、Interpreter呼び出し、後処理較正

結合: /record→推論→保存→/timeline表示のE2E（オフライン/機内モード）

実機: 端末マトリクス（Android 10–15 / iOS 15–18）、安定FPS、温度、消費電力

回帰: モデル更新時に旧モデル対比ABテスト（メトリクス劣化禁止）

12. リスク&対策

データ不均衡: マイノリティ感情の再重み付け・再サンプリング

雑音環境: ノイズ条件の拡張学習＋簡易NR

方言/言語: 多言語追加と逐次ファインチューニング計画

モデル肥大化: 量子化・構造剪定で≤5MB維持

誤推定UX: 低信頼時の安全なUIコピー（励まし・再録案内）

13. 成果物（Deliverables）

emotion_model.tflite（量子化済・バージョンタグ付）

labels.json（クラス順・カラー対応表）

preprocess.py/train.ipynb/export_tflite.py（再現用）

Flutter側EmotionAnalyzer実装（Dart）＋使用例

評価レポート（データ分布・混同行列・F1/ROC・端末計測）

運用ドキュメント（モデル更新手順・ロールバック）
