import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from pathlib import Path
import hologram_utility as h

script_dir = Path(__file__).resolve().parent.parent     #C:\Users\YutoMatsuo\Desktop\Research\debug
model_name = "single_triangle7"
rotation_dir = script_dir/"debug_rotation"
dir_path = script_dir/"debug_npy"/f"{model_name}.npy"

script_path = os.path.abspath(__file__)

#パラメータ
pitch = 4.5e-3
WAVELENGTH = 530e-6
Nx, Ny = 512, 512
X_start, Y_start = Nx // 2, Ny // 2
z = 0

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

def Rotation_matrix_Rodrigues(n):
    # 1. 入力ベクトルの正規化
    norm_n = np.linalg.norm(n)
    if norm_n == 0:
        return np.eye(3) # ゼロベクトルの場合は回転なし（単位行列）
    n = n / norm_n
    
    ez = np.array([0, 0, 1])
    
    # 2. 特異点（n が真上 [0,0,1] または 真下 [0,0,-1] を向いている場合）の処理
    # 外積 k のノルムが 0 になるため、個別に対処が必要
    if np.allclose(n, ez):
        return np.eye(3)  # すでに同じ向きなので回転なし
    if np.allclose(n, -ez):
        # 真下を向いている場合は、x軸（またはy軸）周りに180度回転させる
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]])
    
    # 3. 回転軸 k の算出と正規化
    k = np.cross(ez, n)
    k = k / np.linalg.norm(k)
    
    # 4. 回転角 theta の算出（計算誤差対策の clip を追加）
    theta = np.arccos(np.clip(np.dot(ez, n), -1.0, 1.0))
    
    # 5. 歪対称行列（k_mat）の定義（カンマの追加と np.array 化）
    k_mat = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
    
    # 6. ロドリゲスの公式による回転行列の計算（行列積 @ を使用）
    R = np.eye(3) + k_mat * np.sin(theta) + (k_mat @ k_mat) * (1 - np.cos(theta))
    
    return R

def Hz(z):
    term = 1/WAVELENGTH**2 - U**2 - V**2
    Hz = np.exp(1j * 2* np.pi * z * np.sqrt(term))

    return Hz


g = np.load(dir_path)
plt.figure(1)
plt.imshow(h.normalize("log", g), 'gray')

g_pad = h.padding(g, Ny, Nx)
G = h.FFT(g_pad)
G_recon = G * Hz(z)
g_recon_pad = h.IFFT(G_recon)

g_recon = h.cutting(g_recon_pad, Ny, Nx)

g_view = h.normalize("log", g_recon)

plt.figure(2)
plt.imshow(g_view, 'gray')
plt.show()

"""ホログラム回転運動補償計算"""
for i, deg in enumerate(range(-45, 46)):
    theta = np.radians(deg)
    R_plus = Rotation_matrix(theta) # 回転行列
    U_p, V_p, W_p = U, V, W
    alpha = U_p*R_plus[0, 0] + V_p*R_plus[0, 1] + W_p*R_plus[0, 2] - R_plus[0, 2]/WAVELENGTH
    beta = U_p*R_plus[1, 0] + V_p*R_plus[1, 1] + W_p*R_plus[1, 2] - R_plus[1, 2]/WAVELENGTH
    gamma = U_p*R_plus[2, 0] + V_p*R_plus[2, 1] + W_p*R_plus[2, 2] - R_plus[2, 2]/WAVELENGTH
    jacobian_plus = np.abs(R_plus[0, 0]*R_plus[1, 1] - R_plus[0, 1]*R_plus[1, 0])


    interp = RegularGridInterpolator(
        (v, u),
        G_recon*np.exp(-1j*2*np.pi*z*gamma),
        # method='linear',
        method='cubic',
        bounds_error=False,
        fill_value=0
    )

    query_points = np.stack([beta, alpha], axis=-1)
    G_transfer_rotate = interp(query_points)
    G_recon_rotate = G_transfer_rotate * Hz(-z)

    g_recon_rotate_pad = h.IFFT(G_recon_rotate)
    g_recon_rotate = h.cutting(g_recon_rotate_pad, Ny, Nx)

    model_save_npy = rotation_dir/"interp_rotation_npy"/f"{i:02d}_{model_name}_{deg:+}.npy"
    model_save_bmp = rotation_dir/"interp_rotation_bmp"/f"{i:02d}_{model_name}_{deg:+}.bmp"

    np.save(model_save_npy, g_recon_rotate)
    Image.fromarray(h.normalize("log", g_recon_rotate), mode="L").save(model_save_bmp, format="BMP")

    # plt.figure(3)
    # plt.imshow(h.normalize("log", g_recon_rotate), 'gray')

    # g_recon_rotate_right = h.cutting(h.IFFT(G_recon_rotate * Hz(np.sin(theta))), Ny, Nx)
    # plt.figure(4)
    # plt.imshow(h.normalize("log", g_recon_rotate_right), 'gray')
    # plt.show()

# plt.show()