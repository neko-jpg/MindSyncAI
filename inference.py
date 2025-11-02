import os
import argparse
import numpy as np
import librosa
import tensorflow as tf

# --- 定数定義 ---
MODELS_PATH = 'models/'
TFLITE_MODEL_NAME = 'emotion_model.tflite'
LABELS_NAME = 'labels.txt'

MODEL_PATH = os.path.join(MODELS_PATH, TFLITE_MODEL_NAME)
LABEL_PATH = os.path.join(MODELS_PATH, LABELS_NAME)

def extract_feature(file_name):
    """
    ファイルからMFCC特徴量を抽出する.
    ノートブックのロジックと一貫性を保つ.
    """
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=3, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"特徴量抽出エラー: {file_name} - {e}")
        return None
    return mfccs_processed

def predict(audio_path):
    """
    音声ファイルから感情を推論する.
    """
    # --- 1. モデルとラベルのロード ---
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        print("エラー: モデルファイルまたはラベルファイルが見つかりません。")
        print(f"'{MODEL_PATH}' と '{LABEL_PATH}' を確認してください。")
        print("まず training.ipynb を実行してモデルを生成してください。")
        return

    # ラベルのロード
    with open(LABEL_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # TFLiteモデルのロードとインタープリタの初期化
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # --- 2. 音声データの前処理 ---
    print(f"\\n'{audio_path}' の感情を推論します...")
    feature = extract_feature(audio_path)
    if feature is None:
        return

    # 入力形式をモデルに合わせる (1, features, 1)
    input_data = np.expand_dims(feature, axis=0)
    input_data = np.expand_dims(input_data, axis=2).astype(np.float32)

    # --- 3. 推論の実行 ---
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # --- 4. 結果の表示 ---
    predicted_index = np.argmax(output_data[0])
    predicted_emotion = labels[predicted_index]
    confidence = output_data[0][predicted_index]

    print(f"\\n推論結果: {predicted_emotion} (信頼度: {confidence:.2f})")
    print("-" * 30)
    print("各感情の確率:")
    for i, label in enumerate(labels):
        print(f"- {label}: {output_data[0][i]:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='音声ファイルから感情を推論するスクリプト')
    parser.add_argument('audio_file', type=str, help='推論対象の音声ファイル (.wav)')

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"エラー: 指定されたファイルが見つかりません: {args.audio_file}")
    else:
        predict(args.audio_file)
