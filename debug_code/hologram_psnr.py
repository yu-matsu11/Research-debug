import numpy as np
import os

def calculate_complex_psnr(data1, data2):
    """
    複素データ（data1, data2）の間のPSNRを計算する関数
    """
    # 1. 複素数としての差の絶対値の二乗和からMSEを計算
    # np.abs(data1 - data2)**2 により、実部と虚部を合わせた距離の二乗が計算されます
    mse = np.mean(np.abs(data1 - data2) ** 2)
    
    # 二つのデータが完全に一致している場合、MSEは0になり、PSNRは無限大（または定義不可）になる
    if mse == 0:
        return float('inf')
    
    # 2. データの最大値（ピーク値）を定義
    # 正解データ（data1）の絶対値の最大値を用いるのが一般的です
    max_val = np.max(np.abs(data1))
    
    # 3. PSNRの計算
    psnr = 10 * np.log10((max_val ** 2) / mse)
    
    return psnr

script_path = os.path.abspath(__file__)
debug_dir = os.path.dirname(script_path) #c:\Users\YutoMatsuo\Desktop\Research\debug\debug_code
debug_path = os.path.dirname(debug_dir) #c:\Users\YutoMatsuo\Desktop\Research\debug
data1 = np.load(os.path.join(debug_path, "debug_npy", "single_triangle8_compensation_hologram.npy"))
data2 = np.load(os.path.join(debug_path, "debug_npy", "single_triangle8_rot_hologram.npy"))

psnr_val = calculate_complex_psnr(data1, data2)
print(psnr_val)