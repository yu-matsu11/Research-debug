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
G_recon_view = h.normalize_255("log", G_recon)
plt.figure(1)
plt.imshow(g_view, 'gray')

"""ホログラム回転運動補償計算"""
theta = np.radians(60)
R_plus = Rotation_matrix(theta) # 回転行列
jacobian_plus = np.abs(R_plus[0, 0]*R_plus[1, 1] - R_plus[0, 1]*R_plus[1, 0])
alpha_p = R_plus[0, 0]*(U-R_plus[0, 2]/WAVELENGTH) + R_plus[1, 0]*(V-R_plus[1, 2]/WAVELENGTH) + R_plus[2, 0]*(W)
beta_p = R_plus[0, 1]*(U-R_plus[0, 2]/WAVELENGTH) + R_plus[1, 1]*(V-R_plus[1, 2]/WAVELENGTH) + R_plus[2, 1]*(W)
# gamma_p = np.sqrt(1/WAVELENGTH**2 - alpha_p**2 - beta_p**2)


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
G_recon_rotate_view = h.normalize_255("log", G_recon_rotate)
plt.imshow(g_recon_rotate_view, 'gray')

theta_m = np.radians(-60)
R_minus = Rotation_matrix(theta_m)

# ==========================================
# VOT : pre-transform carrier compensation
# global / parallel -> local / tilted
# ==========================================

# carrier shift
D_u = R_minus[2, 0] / WAVELENGTH
D_v = R_minus[2, 1] / WAVELENGTH

# 先にcarrierを周波数座標へ加える
U_phys = U + D_u
V_phys = V + D_v

# w_t'
term_m = 1 / WAVELENGTH**2 - U_phys**2 - V_phys**2

valid_m = term_m >= 0
W_m = np.zeros_like(term_m)
W_m[valid_m] = np.sqrt(term_m[valid_m])

# VOT-pre の周波数座標変換
# u_s = a1*u_t' + a4*v_t' + a7*w_t'
# v_s = a2*u_t' + a5*v_t' + a8*w_t'
alpha_m = (
    R_minus[0, 0] * U_phys
    + R_minus[0, 1] * V_phys
    + R_minus[0, 2] * W_m
)

beta_m = (
    R_minus[1, 0] * U_phys
    + R_minus[1, 1] * V_phys
    + R_minus[1, 2] * W_m
)


# ==========================================
# Jacobian
# ==========================================

W_safe = np.where(
    valid_m & (W_m != 0),
    W_m,
    np.inf
)

du_dU = (
    R_minus[0, 0]
    - R_minus[0, 2] * U_phys / W_safe
)

du_dV = (
    R_minus[0, 1]
    - R_minus[0, 2] * V_phys / W_safe
)

dv_dU = (
    R_minus[1, 0]
    - R_minus[1, 2] * U_phys / W_safe
)

dv_dV = (
    R_minus[1, 1]
    - R_minus[1, 2] * V_phys / W_safe
)

jacobian_minus = np.abs(
    du_dU * dv_dV - du_dV * dv_dU
)

jacobian_minus[~valid_m] = 0
jacobian_minus[~np.isfinite(jacobian_minus)] = 0


# ==========================================
# Interpolation
# ==========================================

sqrt_J = np.sqrt(jacobian_minus)

interp = RegularGridInterpolator(
    (v, u),
    G_recon_rotate * sqrt_J,
    method='cubic',
    bounds_error=False,
    fill_value=0
)

query_points = np.stack(
    [beta_m, alpha_m],
    axis=-1
)

G_transfer_rotate_m = (
    sqrt_J * interp(query_points)
)

G_recon_rotate_m = G_transfer_rotate_m

# theta_m = np.radians(-60)
# R_minus = Rotation_matrix(theta_m) # 回転行列
# U_p, V_p, W_p = U, V, W
# alpha_m = U_p*R_minus[0, 0] + V_p*R_minus[0, 1] + W_p*R_minus[0, 2] - R_minus[0, 2]/WAVELENGTH
# beta_m = U_p*R_minus[1, 0] + V_p*R_minus[1, 1] + W_p*R_minus[1, 2] - R_minus[1, 2]/WAVELENGTH
# gamma_m = U_p*R_minus[2, 0] + V_p*R_minus[2, 1] + W_p*R_minus[2, 2] - R_minus[2, 2]/WAVELENGTH
# jacobian_minus = np.abs(R_minus[0, 0]*R_minus[1, 1] - R_minus[0, 1]*R_minus[1, 0])


# interp = RegularGridInterpolator(
#     (v, u),
#     G_recon_rotate,
#     # method='linear',
#     method='cubic',
#     bounds_error=False,
#     fill_value=0
# )

# query_points = np.stack([beta_m, alpha_m], axis=-1)
# G_transfer_rotate_m = jacobian_minus * interp(query_points)
# G_recon_rotate_m = G_transfer_rotate_m

# g_recon_rotate_m_pad = h.IFFT(G_recon_rotate_m * Hz(-z))
g_recon_rotate_m_pad = h.IFFT(G_recon_rotate_m)
# g_recon_rotate_m_pad = h.IFFT(G_recon_rotate_m*Hz(-np.sqrt(3)/4))
g_recon_rotate_m = h.cutting(g_recon_rotate_m_pad, Ny, Nx)

plt.figure(3)
g_recon_rotate_m_view = h.normalize_255("log", g_recon_rotate_m)
plt.imshow(g_recon_rotate_m_view, 'gray')

# # Save the initial and +60-degree hologram/spectrum images as 8-bit BMP files.
# save_dir = rotation_dir / "rotation_60deg_bmp"
# save_dir.mkdir(parents=True, exist_ok=True)

# output_images = {
#     "01_initial_hologram.bmp": g_view,
#     "02_initial_spectrum.bmp": G_recon_view,
#     "03_rotated_spectrum_+60deg.bmp": G_recon_rotate_view,
#     "04_rotated_hologram_+60deg.bmp": g_recon_rotate_view,
# }

# for filename, image in output_images.items():
#     Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
#         save_dir / filename,
#         format="BMP",
#     )

# print(f"4 BMP images saved to: {save_dir}")
plt.show()
