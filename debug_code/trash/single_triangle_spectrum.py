import os, trimesh
import numpy as np
from PIL import Image
import matplotlib.path as mpltPath
from scipy.interpolate import RegularGridInterpolator

def RotMatrix(n):
    n = n / np.linalg.norm(n)

    ex, ez = [1, 0, 0], [0, 0, 1]
    cos_theta = ez @ n
    sin_theta = np.linalg.norm(np.cross(ez, n))
    nn = np.array([n[0], n[1], 0.0])
    nn = nn / np.linalg.norm(nn)
    cos_phi = ex @ nn
    sin_phi = np.linalg.norm(np.cross(ex, nn))
    if n[1]<0:
        sin_phi = -sin_phi
    R = np.array([[cos_phi*cos_theta, cos_theta*sin_phi, -sin_theta],
                 [-sin_phi, cos_phi, 0],
                 [cos_phi*sin_theta, sin_phi*sin_theta, cos_theta]])
    if abs(n[2] - 1) < 1e-6:
        R = np.eye(3)
    return R

#パラメータ
pitch = 4.5e-3
WAVELENGTH = 530e-9
Nx, Ny = 1024, 1024

model_name = os.path.join("single_triangle.ply")
data = trimesh.load(model_name)

Global_vertices = data.vertices                                         #三角形の各頂点
Global_faces = data.faces                                               #どの頂点を結ぶかの情報
vGlobal_ertex_normals = data.vertex_normals                             #各頂点の方向を表す
Global_face_normals = data.face_normals                                 #三つの頂点を結んだときにできる三角形の面の方向を表す

n_vector = Global_face_normals[0]                                       #三角形の法線ベクトル計算
R = RotMatrix(n_vector)                                                 #回転行列計算
Global_gravitypoints = np.mean(Global_vertices, axis=0)                 #グローバル座標系での三角形の重心計算
local_vertices = (R @ (Global_vertices - Global_gravitypoints).T).T     #ローカル座標系での三角形の頂点(z座標は0)

"""local座標系の(xhat, yhat)平面への変換"""
local_vertices = local_vertices[:, :2]  #zhat成分は０なのでカット
x_hat_min, y_hat_min = np.min(local_vertices, axis=0)
x_hat_max, y_hat_max = np.max(local_vertices, axis=0)

#ピクセル数の計算
Nx_hat = int(np.ceil((x_hat_max - x_hat_min) / pitch))
Ny_hat = int(np.ceil((y_hat_max - y_hat_min) / pitch))

x_hat = np.linspace(x_hat_min, x_hat_max, Nx_hat)           #x_hat_minからx_hat_maxまでをNx_hatで分割して配列としてx_hatに格納
y_hat = np.linspace(y_hat_min, y_hat_max, Ny_hat)           #y_hat_minからy_hat_maxまでをNy_hatで分割して配列としてy_hatに格納
X_hat, Y_hat = np.meshgrid(x_hat, y_hat, indexing='ij')     #格子点の作成

#三角形の内部判定
points = np.vstack((X_hat.ravel(), Y_hat.ravel())).T
path = mpltPath.Path(local_vertices)
mask = path.contains_points(points).reshape(Nx_hat, Ny_hat)

f = np.zeros((Nx_hat, Ny_hat), dtype=np.uint8)
f[mask] = 1
f = f.T

# #ゼロパディング
f_pad = np.zeros((Ny, Nx), dtype=np.float64)
Nx_hat_origin, Ny_hat_origin = Nx // 2, Ny // 2
x_start, y_start = (Nx-Nx_hat) // 2, (Ny-Ny_hat) // 2
f_pad[y_start : y_start + Ny_hat, x_start : x_start + Nx_hat] = f

#周波数座標系(u_hat, v_hat)の作成
Nv_hat, Nu_hat = f_pad.shape
u_hat = np.fft.fftshift(np.fft.fftfreq(Nu_hat, d=pitch))
v_hat = np.fft.fftshift(np.fft.fftfreq(Nv_hat, d=pitch))
U_hat, V_hat = np.meshgrid(u_hat, v_hat)

#ゼロパディング前のローカル座標系での三角形の重心計算
Local_gravitypoints = np.mean(local_vertices, axis=0)
x_hat_origin, y_hat_origin = Local_gravitypoints

#ラスタライズしたときの原点がローカル座標系での座標値の計算
O_x_hat = (x_hat_max - x_hat_min)/2
O_y_hat = (y_hat_max - y_hat_min)/2

# 位相シフト計算
phase_shift = np.exp(1j * 2 * np.pi * ((O_x_hat-x_hat_origin)*U_hat + (O_y_hat-y_hat_origin)*V_hat))

# 単一三角形のローカル座標系でのスペクトル計算
# F = np.fft.fft2(f_pad)
F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f_pad)))
F = F * phase_shift

# スペクトルのゼロパディング
F_pad = np.zeros((Ny*2, Nx*2), dtype=np.complex128)
X_start, Y_start = Nx // 2, Ny // 2
F_pad[Y_start : Y_start + Ny, X_start : X_start + Nx] = F

F_view = np.log(np.abs(F_pad) + 1)
spectrum_view = (255 * (F_view - F_view.min()) / (F_view.max() - F_view.min())).astype(np.uint8)
Image.fromarray(spectrum_view, mode="L").save("single_triangle_spectrum.bmp", format="BMP")