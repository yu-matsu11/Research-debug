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

g_view = h.normalize_255("log", g)
plt.figure(1)
plt.imshow(g_view, 'gray')

"""ホログラム回転運動補償計算"""
theta = np.radians(60)
R_plus = Rotation_matrix(theta) # 回転行列
U_p, V_p, W_p = U, V, W
jacobian_plus = np.abs(R_plus[0, 0]*R_plus[1, 1] - R_plus[0, 1]*R_plus[1, 0])
alpha_p = R_plus[0, 0]*(U_p-R_plus[0, 2]/WAVELENGTH) + R_plus[0, 1]*(V_p-R_plus[1, 2]/WAVELENGTH) + R_plus[0, 2]*(W_p-R_plus[2, 2]/WAVELENGTH)
beta_p = R_plus[1, 0]*(U_p-R_plus[0, 2]/WAVELENGTH) + R_plus[1, 1]*(V_p-R_plus[1, 2]/WAVELENGTH) + R_plus[1, 2]*(W_p-R_plus[2, 2]/WAVELENGTH)
gamma_p = np.sqrt(1/WAVELENGTH**2 - alpha_p**2 - beta_p**2)


interp = RegularGridInterpolator(
    (v, u),
    # G_recon*np.exp(-1j*2*np.pi*z*gamma_p),
    G_recon,
    # method='linear',
    method='cubic',
    bounds_error=False,
    fill_value=0
)

query_points = np.stack([beta_p, alpha_p], axis=-1)
G_transfer_rotate = jacobian_plus * interp(query_points)
G_recon_rotate = G_transfer_rotate

g_recon_rotate_pad = h.IFFT(G_recon_rotate * Hz(-z))
g_recon_rotate = h.cutting(g_recon_rotate_pad, Ny, Nx)

plt.figure(2)
g_recon_rotate_view = h.normalize_255("log", g_recon_rotate)
plt.imshow(g_recon_rotate_view, 'gray')

plt.show()

theta_m = np.radians(60)
R_minus = Rotation_matrix(theta_m) # 回転行列
alpha_m = R_minus[0, 0]*alpha_p + R_minus[0, 1]*beta_p + R_minus[0, 2]*gamma_p - R_minus[0, 2]/WAVELENGTH
beta_m = R_minus[1, 0]*alpha_p + R_minus[1, 1]*beta_p + R_minus[1, 2]*gamma_p - R_minus[1, 2]/WAVELENGTH
gamma_m = np.sqrt(1/WAVELENGTH**2 - alpha_m**2 - beta_m**2)

interp = RegularGridInterpolator(
    (v, u),
    G_recon_rotate,
    method='cubic',
    bounds_error=False,
    fill_value=0
)

query_points = np.stack([beta_m, alpha_m], axis=-1)
G_transfer_rotate_back = interp(query_points)

g_recon_rotate_back_pad = h.IFFT(G_transfer_rotate_back * Hz(-z))
g_recon_rotate_back = h.cutting(g_recon_rotate_back_pad, Ny, Nx)

plt.figure(3)
g_recon_rotate_back_view = h.normalize_255("log", g_recon_rotate_back)
plt.imshow(g_recon_rotate_back_view, 'gray')

plt.show()