import numpy as np

def FFT(X):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(X)))

def IFFT(X):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

def padding(X, Ny, Nx):
    return np.pad(X, ((Ny//2, Ny//2), (Nx//2, Nx//2)), 'constant')

def cutting(X, Ny, Nx):
    Y_start, X_start = Ny // 2, Nx // 2
    return X[Y_start : Y_start+Ny, X_start : X_start+Nx]

def normalize_255(mode, X):

    if mode == "abs":
        X_abs = np.abs(X)
        if X_abs.max() > 0:
            return (255 * (X_abs / X_abs.max())).astype(np.uint8)
        else:
            return np.zeros_like(X_abs, dtype=np.uint8)

    if mode == "angle":
        X_angle = np.angle(X)
        if X_angle.max() > 0:
            return (255 * (X_angle / X_angle.max())).astype(np.uint8)
        else:
            return np.zeros_like(X_angle, dtype=np.uint8)

    if mode == "log":
        X_log = np.log(np.abs(X)+1)
        if X_log.max() > 0:
            return (255 * (X_log / X_log.max())).astype(np.uint8)
        else:
            return np.zeros_like(X_log, dtype=np.uint8)

    if mode == "real":
        X_real = np.real(X)
        if X_real.max() > 0:
            return (255 * (X_real / X_real.max())).astype(np.uint8)
        else:
            return np.zeros_like(X_real, dtype=np.uint8)
        
def normalize_zero_to_one(mode, X):
    # 1. 各モードに応じてベースとなる実数データを抽出
    if mode == "abs":
        X_target = np.abs(X)
    elif mode == "angle":
        X_target = np.angle(X)
    elif mode == "log":
        X_target = np.log(np.abs(X) + 1.0)
    elif mode == "real":
        X_target = np.real(X)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2. 最小値と最大値を取得
    x_min = np.min(X_target)
    x_max = np.max(X_target)
    
    # 3. 最大値と最小値の差（データの幅）を計算
    x_range = x_max - x_min
    
    # 全て同じ値（差が0）の場合は、ゼロで埋めた配列を返す（ゼロ除算対策）
    if x_range == 0:
        return np.zeros_like(X_target, dtype=np.float64)
    
    # 4. 確実に 0.0 ~ 1.0 の範囲にスケーリング
    return (X_target - x_min) / x_range