import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
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

g = np.load(dir_path)                   # 複素データの読み込み

g_pad = h.padding(g, Ny, Nx)            # パディング
G = h.FFT(g_pad)                        # フーリエ変換
G_recon = G * Hz(z)                     # 伝搬
g_recon_pad = h.IFFT(G_recon*Hz(-np.sqrt(3)/2))           # 逆フーリエ変換

g_recon = h.cutting(g_recon_pad, Ny, Nx)# パディングを切り取る

g_view = h.normalize_255("log", g_recon)# 256階調に正規化して表示用に変換

plt.figure(1)
plt.imshow(g_view, "gray")


theta_m = np.radians(70)
R_minus = Rotation_matrix(theta_m) # 回転行列

D_u = R_minus[2, 0] / WAVELENGTH
D_v = R_minus[2, 1] / WAVELENGTH

U_phys = U + D_u
V_phys = V + D_v

W_phys = np.sqrt(1/WAVELENGTH**2 - U_phys**2 - V_phys**2)

alpha_m = U_phys*R_minus[0, 0] + V_phys*R_minus[0, 1] + W_phys*R_minus[0, 2]
beta_m = U_phys*R_minus[1, 0] + V_phys*R_minus[1, 1] + W_phys*R_minus[1, 2]

du_dU = R_minus[0, 0] - R_minus[0, 2] * (U_phys / W_phys)
du_dV = R_minus[0, 1] - R_minus[0, 2] * (V_phys / W_phys)
dv_dU = R_minus[1, 0] - R_minus[1, 2] * (U_phys / W_phys)
dv_dV = R_minus[1, 1] - R_minus[1, 2] * (V_phys / W_phys)
jacobian_minus = np.abs(du_dU * dv_dV - du_dV * dv_dU)


interp = RegularGridInterpolator(
    (v, u),
    G_recon,
    # method='linear',
    method='cubic',
    bounds_error=False,
    fill_value=0
)

query_points = np.stack([beta_m, alpha_m], axis=-1)
G_transfer_rotate_m = interp(query_points) * jacobian_minus
# G_recon_rotate_m = G_transfer_rotate_m * Hz(-1/2)
G_recon_rotate_m = G_transfer_rotate_m

g_recon_rotate_m_pad = h.IFFT(G_recon_rotate_m)
g_recon_rotate_m = h.cutting(g_recon_rotate_m_pad, Ny, Nx)

g_view_m = h.normalize_255("log", g_recon_rotate_m)
plt.figure(2)
plt.imshow(g_view_m, "gray")
plt.show()
