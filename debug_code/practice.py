import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

#パラメータ
pitch = 4.5e-3
WAVELENGTH = 530e-6
Nx, Ny = 1980, 1080
X_start, Y_start = Nx // 2, Ny // 2
# z = 0
theta = np.radians(60)
phi = np.radians(0)

# 回転行列
Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
              [0, 1, 0],
              [-np.sin(theta), 0, np.cos(theta)]])
# Rx = np.array([[1, 0, 0],
#               [0, np.cos(phi), -np.sin(phi)],
#               [0, np.sin(phi), np.cos(phi)]])

# R = np.dot(Rx, Ry)
R = Ry


# グローバル座標系の周波数座標系
u = np.fft.fftshift(np.fft.fftfreq(Nx, d=pitch))
v = -np.fft.fftshift(np.fft.fftfreq(Ny, d=pitch))
U, V = np.meshgrid(u, v)
pre = U**2 + V**2
W = np.sqrt(1//WAVELENGTH**2 - pre)

script_path = os.path.abspath(__file__)
debug_dir = os.path.dirname(script_path) #c:\Users\YutoMatsuo\Desktop\Research\debug\debug_code
debug_path = os.path.dirname(debug_dir) #c:\Users\YutoMatsuo\Desktop\Research\debug
# dir_path = os.path.join(debug_path, "debug_bmp", "triple_triangle_.bmp")
# dir_path = os.path.join(debug_path, "debug_bmp", "texbunny3000_reconstruction.bmp")
dir_path = os.path.join(debug_path, "debug_bmp", "single_triangle5_reconstruction.bmp")
# dir_path = os.path.join(debug_path, "debug_bmp", "texbunny.bmp")

ha = np.array(Image.open(dir_path).convert("L"), dtype=np.float64)
g = ha/255.0

plt.figure(1)
plt.imshow(g, 'gray')

# g_pad = np.pad(g, ((Ny//2, Ny//2), (Nx//2, Nx//2)), 'constant')
g_recon_p = np.zeros((Ny, Nx), dtype = np.complex128)

term = 1/WAVELENGTH**2 - U**2 - V**2

G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g)))

U_p, V_p, W_p = U, V, W
alpha = U_p*R[0, 0] + V_p*R[0, 1] + W_p*R[0, 2] - R[0, 2]/WAVELENGTH
beta  = U_p*R[1, 0] + V_p*R[1, 1] + W_p*R[1, 2] - R[1, 2]/WAVELENGTH
query_points = np.stack([beta, alpha], axis=-1)

# for z in np.linspace(0.12, 0.15, 4):
for z in range(1):
    # 伝達関数
    Hz = np.exp(1j * 2* np.pi * z * np.sqrt(term))
    G_recon = G * Hz

    interp = RegularGridInterpolator(
        (v, u),
        G_recon,
        # method='linear',
        method='cubic',
        bounds_error=False,
        fill_value=0
    )

    G_recon_p = interp(query_points) * np.abs(R[0, 0]*R[1, 1] - R[0, 1]*R[1, 0])
    g_recon_p += np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_recon_p)))

# g_recon_p = g_recon_p[Y_start : Y_start+Ny, X_start : X_start+Nx]

# plt.figure(3)
# plt.imshow(np.log(np.abs(G_recon)+1), 'gray')
# plt.figure(4)
# plt.imshow(np.log(np.abs(G_recon_p)+1), 'gray')

plt.figure(5)
plt.imshow(np.log(np.abs(g_recon_p)+1), 'gray')

plt.show()