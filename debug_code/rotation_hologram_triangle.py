import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from pathlib import Path
import hologram_utility as h

method_name = ["linear", "cubic"]

script_dir = os.path.abspath(__file__)
research_dir = os.path.dirname(script_dir)
research_path = os.path.dirname(research_dir)

model_name = "single_triangle7"
model_path = os.path.join(research_path, "debug_ply", f"{model_name}.ply")
model_npy = os.path.join(research_path, "debug_experiment", "linear", "single_triangle", "original", "npy")   
model_bmp = os.path.join(research_path, "debug_experiment", "linear", "single_triangle", "original", "bmp")

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

def Hz(z):
    term = 1/WAVELENGTH**2 - U**2 - V**2
    Hz = np.exp(1j * 2* np.pi * z * np.sqrt(term))

    return Hz

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
for n, deg in enumerate(range(-45, 46)):
    theta = np.radians(deg)
    R_plus = h.Rotation_matrix(theta) # 回転行列
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

    holo_save_npy = os.path.join(model_npy, f"{n:03d}_{deg:+03d}.npy")
    holo_save_bmp = os.path.join(model_bmp, f"{n:03d}_{deg:+03d}.bmp")

    np.save(holo_save_npy, g_recon_rotate)
    Image.fromarray(h.normalize("log", g_recon_rotate), mode="L").save(holo_save_bmp, format="BMP")

    # plt.figure(3)
    # plt.imshow(h.normalize("log", g_recon_rotate), 'gray')

    # g_recon_rotate_right = h.cutting(h.IFFT(G_recon_rotate * Hz(np.sin(theta))), Ny, Nx)
    # plt.figure(4)
    # plt.imshow(h.normalize("log", g_recon_rotate_right), 'gray')
    # plt.show()

# plt.show()