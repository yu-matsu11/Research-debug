import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

#パラメータ
pitch = 4.5e-3
WAVELENGTH = 530e-6
Nx, Ny = 1920, 1080
z = 100

# グローバル座標系の周波数座標系
u = np.fft.fftshift(np.fft.fftfreq(Nx*2, d=pitch))
v = -np.fft.fftshift(np.fft.fftfreq(Ny*2, d=pitch))
U, V = np.meshgrid(u, v)
pre = U**2 + V**2

script_path = os.path.abspath(__file__)
debug_dir = os.path.dirname(script_path) #c:\Users\YutoMatsuo\Desktop\Research\debug\debug_code
debug_path = os.path.dirname(debug_dir) #c:\Users\YutoMatsuo\Desktop\Research\debug
dir_path = os.path.join(debug_path, "debug_npy", "triple_triangle3_hologram.npy")
# dir_path = os.path.join(debug_path, "debug_bmp", "single_triangle6_hologram.bmp")
# dir_path = os.path.join(debug_path, "debug_bmp", "texbunny.bmp")

g = np.load(dir_path)

plt.figure(1)
plt.imshow(np.log(np.abs(g)+1), 'gray')

g_pad = np.pad(g, ((Ny//2, Ny//2), (Nx//2, Nx//2)), 'constant')

# 伝達関数
term = 1/WAVELENGTH**2 - U**2 - V**2
Hz = np.exp(1j * 2* np.pi * z * np.sqrt(term))

G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g_pad)))
G_recon = G * Hz
print(np.mean(G_recon))
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
plt.show()