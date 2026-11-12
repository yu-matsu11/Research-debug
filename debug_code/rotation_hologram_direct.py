import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import hologram_utility as h

script_dir = Path(__file__).resolve().parent.parent     #C:\Users\YutoMatsuo\Desktop\Research\debug
model_name = "single_triangle10_rot_hologram"
rotation_dir = script_dir/"debug_rotation"
dir_path = script_dir/"debug_npy"/f"{model_name}.npy"

script_path = os.path.abspath(__file__)

#パラメータ
pitch = 4.5e-3
WAVELENGTH = 530e-6
Nx, Ny = 512, 512
X_start, Y_start = Nx // 2, Ny // 2
z = 2

# グローバル座標系の周波数座標系
u = np.fft.fftshift(np.fft.fftfreq(Nx*2, d=pitch))
v = -np.fft.fftshift(np.fft.fftfreq(Ny*2, d=pitch))
U, V = np.meshgrid(u, v)
pre = U**2 + V**2
W = np.sqrt(1/WAVELENGTH**2 - pre)

def Rotation_matrix(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
              [0, 1, 0],
              [-np.sin(theta), 0, np.cos(theta)]])
    return R

def Hz(z):
    term = 1/WAVELENGTH**2 - U**2 - V**2
    Hz = np.exp(1j * 2* np.pi * z * np.sqrt(term))

    return Hz


g = np.load(dir_path)

g_pad = h.padding(g, Ny, Nx)
G = h.FFT(g_pad)
G_recon = G * Hz(z)
g_recon_pad = h.IFFT(G_recon)

g_recon = h.cutting(g_recon_pad, Ny, Nx)

g_view = h.normalize_255("log", g_recon)

plt.figure(1)
plt.imshow(g_view, 'gray')

"""ホログラム回転運動補償計算：p1 -> p2 の直接変換"""

# 各平面のグローバル座標系に対する角度
theta_1 = np.radians(0)  # p1の角度
theta_2 = np.radians(-30)  # p2の角度

R1 = Rotation_matrix(theta_1)
R2 = Rotation_matrix(theta_2)

# p2座標からp1座標への相対回転行列
R12 = R1.T @ R2

# p2側のキャリアシフト
U2_phys = U + R2[2, 0] / WAVELENGTH
V2_phys = V + R2[2, 1] / WAVELENGTH

# p2側の縦方向周波数
W2_squared = (
    1 / WAVELENGTH**2
    - U2_phys**2
    - V2_phys**2
)

valid = W2_squared > 0

W2_phys = np.zeros_like(U)
W2_phys[valid] = np.sqrt(W2_squared[valid])

# p2 -> p1 の周波数座標変換
alpha = (
    R12[0, 0] * U2_phys
    + R12[0, 1] * V2_phys
    + R12[0, 2] * W2_phys
    - R1[2, 0] / WAVELENGTH
)

beta = (
    R12[1, 0] * U2_phys
    + R12[1, 1] * V2_phys
    + R12[1, 2] * W2_phys
    - R1[2, 1] / WAVELENGTH
)

# 入力側も正の縦方向周波数の枝に限定する。
W1_phys = (
    R12[2, 0] * U2_phys
    + R12[2, 1] * V2_phys
    + R12[2, 2] * W2_phys
)
valid &= W1_phys > 0

# ヤコビアンは微分行列の行列式から |w1' / w2'| となる。
# 添付資料の簡約式のマイナスはプラスが正しい。
# b4*b8 - b7*b5 = b3, b7*b2 - b1*b8 = b6。
jacobian = np.zeros_like(U)
jacobian[valid] = np.abs(W1_phys[valid] / W2_phys[valid])

# p1の角スペクトルを補間
interp = RegularGridInterpolator(
    (v, u),
    G_recon,
    method="cubic",
    bounds_error=False,
    fill_value=0
)

query_points = np.stack([beta[valid], alpha[valid]], axis=-1)

# 1回の補間でp2の角スペクトルを生成
G_p2 = np.zeros_like(G_recon, dtype=np.complex128)
G_p2[valid] = interp(query_points) * jacobian[valid]

# 逆フーリエ変換
g_p2_pad = h.IFFT(G_p2)
g_p2 = h.cutting(g_p2_pad, Ny, Nx)

# 表示
plt.figure(2)
plt.imshow(h.normalize_255("log", g_p2), "gray")
plt.title("Direct VOT: p1 -> p2")

plt.show()
