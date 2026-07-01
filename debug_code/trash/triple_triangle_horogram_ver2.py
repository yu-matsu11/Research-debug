import os, trimesh
import numpy as np
from PIL import Image
import matplotlib.path as mpltPath
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def RotMatrix(n):
    n = n / np.linalg.norm(n)

    if abs(n[2] - 1.0) < 1e-9:
        return np.eye(3)

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
double_pitch = 2 * pitch
WAVELENGTH = 530e-6
Nx, Ny = 1980, 1080
Lx, Ly = Nx*pitch, Ny*pitch

# グローバル座標系の周波数座標系
u = np.linspace(-1/double_pitch, (1/double_pitch - 1/(2*Lx)), Nx, endpoint=True)
v = np.linspace(1/double_pitch, (-1/double_pitch + 1/(2*Ly)), Ny, endpoint=True)
U, V = np.meshgrid(u, v)
W = np.sqrt(1/WAVELENGTH**2 - U**2 - V**2)
G_total = np.zeros((Ny, Nx), dtype=np.complex128)

model_name = os.path.join("triple_triangle2.ply")
data = trimesh.load(model_name)

Global_vertices = data.vertices                                         #三角形の各頂点
Global_faces = data.faces                                               #どの頂点を結ぶかの情報
Global_vertex_normals = data.vertex_normals                             #各頂点の方向を表す
Global_face_normals = data.face_normals                                 #三つの頂点を結んだときにできる三角形の面の方向を表す

for i in range(1):
# i = 0

    face_indices = Global_faces[i]
    current_face_vertices = Global_vertices[face_indices]

    n_vector = Global_face_normals[i]                                       #三角形の法線ベクトル計算
    R = RotMatrix(n_vector)                                                 #回転行列計算
    Global_gravitypoints = np.mean(Global_vertices, axis=0)                 #グローバル座標系での三角形の重心計算
    local_vertices = (R @ (Global_vertices - Global_gravitypoints).T).T     #ローカル座標系での三角形の頂点(z座標は0)

    """local座標系の(xhat, yhat)平面への変換"""
    local_vertices = local_vertices[:, :2]  #zhat成分は０なのでカット
    x_hat_min, y_hat_min = np.min(local_vertices, axis=0)
    x_hat_max, y_hat_max = np.max(local_vertices, axis=0)

    #ピクセル数の計算
    Nx_hat = int(np.ceil((x_hat_max - x_hat_min) / pitch))  #Nx_p
    Nx_hat += Nx_hat%2
    Ny_hat = int(np.ceil((y_hat_max - y_hat_min) / pitch))  #Ny_p
    Ny_hat += Ny_hat%2

    Lx_hat , Ly_hat = Nx_hat*pitch, Ny_hat*pitch # Lx_p, Ly_p

    x_hat = np.linspace(-Lx_hat/2, (Lx_hat/2 - pitch), Nx_hat)           #x_hat_minからx_hat_maxまでをNx_hatで分割して配列としてx_hatに格納
    y_hat = np.linspace(Ly_hat/2, (-Ly_hat/2 + pitch), Ny_hat)           #y_hat_minからy_hat_maxまでをNy_hatで分割して配列としてy_hatに格納
    X_hat, Y_hat = np.meshgrid(x_hat, y_hat, indexing='ij')     #格子点の作成

    #三角形の内部判定
    points = np.vstack((X_hat.ravel(), Y_hat.ravel())).T
    path = mpltPath.Path(local_vertices)
    mask = path.contains_points(points).reshape(Nx_hat, Ny_hat)

    f = np.zeros((Nx_hat, Ny_hat), dtype=np.uint8)
    f[mask] = 1
    f = f.T

    # ゼロパディング
    f_pad = np.pad(f, ((Ny_hat//2, Ny_hat//2), (Nx_hat//2, Nx_hat//2)), 'constant')

    #ゼロパディング前のローカル座標系での三角形の重心計算
    Local_gravitypoints = np.mean(local_vertices, axis=0)
    x_hat_origin, y_hat_origin = Local_gravitypoints

    #ラスタライズしたときの原点がローカル座標系での座標値の計算
    O_x_hat = (x_hat_max - x_hat_min)/2     # Lx_p/2
    O_y_hat = (y_hat_max - y_hat_min)/2     # Ly_p/2

    x_hat_sft = O_x_hat - np.max(local_vertices[:, 0])
    y_hat_sft = O_y_hat - np.max(local_vertices[:, 1])

    #周波数座標系(u_hat, v_hat)の作成
    u_hat = np.linspace(-1/double_pitch, (1/double_pitch - 1/(Lx_hat*2)), int(Nx_hat*2))    #fx_pb
    v_hat = np.linspace(1/double_pitch, (-1/double_pitch + 1/(Ly_hat*2)), int(Ny_hat*2))    #fy_pb
    U_hat, V_hat = np.meshgrid(u_hat, v_hat)

    # 位相シフト計算
    phase_shift = np.exp(1j * 2 * np.pi * (x_hat_sft*U_hat + y_hat_sft*V_hat))

    # 単一三角形のローカル座標系でのスペクトル計算
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f_pad)))
    F = F * phase_shift

    #補間法
    interp = RegularGridInterpolator(
        (V_hat[:, 0], U_hat[0, :]),
        F,
        method='linear',
        bounds_error=False,
        fill_value=0
    )

    x_interp_u = R[0, 0]*U + R[0, 1]*V + R[0, 2]*W - R[0, 2]/WAVELENGTH
    y_interp_v = R[1, 0]*U + R[1, 1]*V + R[1, 2]*W - R[1, 2]/WAVELENGTH

    query_points = np.stack((y_interp_v.ravel(), x_interp_u.ravel()), axis=-1)

    F_interp = interp(query_points).reshape(x_interp_u.shape)                     #これが違う

    #ヤコビアン行列計算
    Jr = R[0, 0]*R[1, 1] - R[0, 1]*R[1, 0]                                  #これはあってる
    x_c, y_c, z_c = Global_gravitypoints                                    #これはあってる
    E1 = np.exp(-1j * 2 * np.pi * (U*x_c + V*y_c + W*z_c - z_c/WAVELENGTH)) #これはあってない
    # グローバル座標系での三角形のスペクトル
    G = Jr * E1 * F_interp

    G_total += G

# hologram
g = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_total)))

plt.figure(1)
plt.imshow(np.log(np.abs(g)), 'gray')
plt.show()