import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

#パラメータ
pitch = 4.5e-3
WAVELENGTH = 530e-6
Nx, Ny = 1024, 1024
x, y = 1, 1
z = 0

# グローバル座標系の周波数座標系
u = np.fft.fftshift(np.fft.fftfreq(Nx*2, d=pitch))
v = -np.fft.fftshift(np.fft.fftfreq(Ny*2, d=pitch))
U, V = np.meshgrid(u, v)
pre = U**2 + V**2

script_dir = Path(__file__).resolve().parent.parent
model_name = "single_triangle5"
dir_path = script_dir/"debug_npy"/f"{model_name}.npy"


ha = np.load(dir_path)
g = np.log(np.abs(ha)+1)

plt.figure(1)
plt.imshow(g, 'gray')
plt.show()

g_pad = np.pad(g, ((Ny//2, Ny//2), (Nx//2, Nx//2)), 'constant')

# 伝達関数
term = 1/WAVELENGTH**2 - U**2 - V**2
Hz = np.exp(1j * 2* np.pi * z * np.sqrt(term))

G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g_pad)))
G_recon = G * Hz
g_recon_pad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_recon)))

X_start, Y_start = Nx // 2, Ny // 2
g_recon = g_recon_pad[Y_start : Y_start+Ny, X_start : X_start+Nx]

g_recon_abs = np.abs(g_recon)
if g_recon_abs.max() > 0:
    g_view = (255 * (g_recon_abs / g_recon_abs.max())).astype(np.uint8)
else:
    g_view = np.zeros_like(g_recon_abs, dtype=np.uint8)

plt.figure(2)
plt.imshow(g_view, 'gray')

# for i in np.arange(0.0, 1.05, 0.1):
for i in range(1):
    # G_motion = G_recon * np.exp(1j * 2 * np.pi * (i*U+y*V))
    G_motion = G_recon * np.exp(-1j * 2 * np.pi * (x*U+y*V))
    g_motion_pad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_motion)))
    g_motion = g_motion_pad[Y_start : Y_start+Ny, X_start : X_start+Nx]

    g_recon_abs = np.abs(g_motion)
    if g_recon_abs.max() > 0:
        g_view = (255 * (g_recon_abs / g_recon_abs.max())).astype(np.uint8)
    else:
        g_view = np.zeros_like(g_recon_abs, dtype=np.uint8)

    z = i*10
    plt.figure(z)
    plt.imshow(g_view, 'gray')
plt.show()