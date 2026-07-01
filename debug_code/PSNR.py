import os
import re
import csv
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import hologram_utility as h  # 既存のユーティリティをインポート

def calculate_psnr(ref_img, test_img):
    """
    h.normalizeによって0~1に正規化された画像間のPSNRを計算する関数
    """
    mse = np.mean((ref_img.astype(np.float64) - test_img.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr_val = peak_signal_noise_ratio(ref_img, test_img, data_range=1.0)
    return psnr_val

def calculate_ssim(ref_img, test_img):
    """
    h.normalizeによって0~1に正規化された画像間のSSIMを計算する関数
    """
    ssim_val = structural_similarity(ref_img, test_img, data_range=1.0)
    return ssim_val


# --- メイン処理 ---

# フォルダパスの設定
base_dir = Path(r"C:\Users\YutoMatsuo\Desktop\Research\debug\debug_rotation")
ref_dir = base_dir / "hologram_rotation_npy"  # 所望のデータ（基準）
test_dir = base_dir / "interp_rotation_npy"  # 補間データ（比較対象）

# CSVファイルの保存先パス
csv_save_path = base_dir / "psnr_ssim_results.csv"

# 所望のデータフォルダからすべての .npy ファイルを取得してソート
ref_files = sorted(ref_dir.glob("*.npy"))

# CSVファイルへの書き込み処理を開始
with open(csv_save_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # 💡 1. ヘッダー（列の名前）を書き込む
    writer.writerow(["File Name", "Angle", "PSNR (dB)", "SSIM"])
    
    print(f"{'ファイル名':<30} | {'角度':<10} | {'PSNR (dB)':<10} | {'SSIM':<6}")
    print("-" * 65)
    
    # 💡 2. 各ファイルをループ処理して計算・保存
    for ref_path in ref_files:
        file_name = ref_path.name
        test_path = test_dir / file_name
        
        # 配下に対応する補間ファイルがあるかチェック
        if not test_path.exists():
            print(f"Warning: {file_name} が補間フォルダに見つかりません。スキップします。")
            continue
            
        # 💡 ファイル名から角度（例: "-45", "+15" などの数値）を自動で抽出する
        # ファイル名の末尾（.npyの手前）にある「ハイフンかプラスで始まる数字」を検索します
        match = re.search(r"([+-]?\d+)(?=\.npy)", file_name)
        if match:
            deg = int(match.group(1))
        else:
            deg = "Unknown"  # 万が一解析できなかった場合のフォールバック
            
        # データの読み込み
        ref_data = np.load(ref_path)
        test_data = np.load(test_path)
        
        # 💡 自作の関数を使って 0~1 (float64) に正規化
        # ホログラムの振幅強度を比較するため、ここでは "abs" モードに指定しています
        ref_norm = h.normalize_zero_to_one("abs", ref_data)
        test_norm = h.normalize_zero_to_one("abs", test_data)
        
        # PSNR と SSIM の計算
        psnr_score = calculate_psnr(ref_norm, test_norm)
        ssim_score = calculate_ssim(ref_norm, test_norm)
        
        # ターミナルへ表示
        print(f"{file_name:<30} | {deg:<10} | {psnr_score:>10.4f} | {ssim_score:>6.4f}")

        # 💡 変数 deg（抽出した角度）を含めてCSVに書き込む
        writer.writerow([file_name, deg, f"{psnr_score:.4f}", f"{ssim_score:.4f}"])

print("-" * 65)
print(f"結果は次のパスに保存されました: {csv_save_path}")
