import re
import csv
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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


def process_method(method_dir):
    ref_dir = method_dir / "single_triangle" / "original" / "npy"
    test_dir = method_dir / "single_triangle" / "interpolation" / "npy"
    csv_save_path = method_dir / f"psnr_ssim_{method_dir.name}_results.csv"

    ref_files = sorted(ref_dir.glob("*.npy"))
    test_names = {path.name for path in test_dir.glob("*.npy")}
    ref_names = {path.name for path in ref_files}
    if ref_names != test_names:
        missing = sorted(ref_names - test_names)
        extra = sorted(test_names - ref_names)
        raise ValueError(
            f"File names do not match in {method_dir.name}: "
            f"missing={missing}, extra={extra}"
        )

    with open(csv_save_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Angle", "PSNR (dB)", "SSIM"])

        print(f"\n[{method_dir.name}]")
        print(f"{'ファイル名':<30} | {'角度':<10} | {'PSNR (dB)':<10} | {'SSIM':<6}")
        print("-" * 65)

        for ref_path in ref_files:
            file_name = ref_path.name
            test_path = test_dir / file_name
            match = re.search(r"([+-]?\d+)(?=\.npy)", file_name)
            deg = int(match.group(1)) if match else "Unknown"

            ref_data = np.load(ref_path)
            test_data = np.load(test_path)
            if ref_data.shape != test_data.shape:
                raise ValueError(
                    f"Shape mismatch: {file_name}: "
                    f"original={ref_data.shape}, interpolation={test_data.shape}"
                )

            ref_abs = np.abs(ref_data).astype(np.float64)
            test_abs = np.abs(test_data).astype(np.float64)
            ref_min = np.min(ref_abs)
            ref_range = np.max(ref_abs) - ref_min
            if ref_range == 0:
                raise ValueError(f"Reference data has no dynamic range: {ref_path}")

            ref_norm = (ref_abs - ref_min) / ref_range
            test_norm = (test_abs - ref_min) / ref_range
            psnr_score = calculate_psnr(ref_norm, test_norm)
            ssim_score = calculate_ssim(ref_norm, test_norm)

            print(
                f"{file_name:<30} | {deg:<10} | "
                f"{psnr_score:>10.4f} | {ssim_score:>6.4f}"
            )
            writer.writerow(
                [file_name, deg, f"{psnr_score:.4f}", f"{ssim_score:.4f}"]
            )

    print(f"結果を保存しました: {csv_save_path}")


# --- メイン処理 ---

base_dir = Path(__file__).resolve().parent.parent / "debug_experiment"
for method in ("linear", "cubic"):
    process_method(base_dir / method)
