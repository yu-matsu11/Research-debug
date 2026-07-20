import os, trimesh
import numpy as np
from PIL import Image
import matplotlib.path as mpltPath
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
import matplotlib.pyplot as plt
from pathlib import Path
import hologram_utility as h

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
WAVELENGTH = 530e-6
Nx, Ny = 512, 512

# グローバル座標系の周波数座標系
u = np.fft.fftshift(np.fft.fftfreq(Nx*2, d=pitch))
v = -np.fft.fftshift(np.fft.fftfreq(Ny*2, d=pitch))
U, V = np.meshgrid(u, v)
pre = U**2 + V**2
G_total = np.zeros((Ny*2, Nx*2), dtype=np.complex128)

# method_name = ["linear", "cubic"]
method_name = ["cubic"]

script_dir = os.path.abspath(__file__)
research_dir = os.path.dirname(script_dir)
research_path = os.path.dirname(research_dir)

model_name = "single_triangle7"
model_path = os.path.join(research_path, "debug_ply", f"{model_name}.ply")


data = trimesh.load(model_path)

Global_vertices = data.vertices                                         #三角形の各頂点
Global_faces = data.faces                                               #どの頂点を結ぶかの情報
Global_vertex_normals = data.vertex_normals                             #各頂点の方向を表す
Global_face_normals = data.face_normals                                 #三つの頂点を結んだときにできる三角形の面の方向を表す

for method in method_name:

    model_npy = os.path.join(research_path, "debug_experiment", method, "single_triangle", "original", "npy")   
    model_bmp = os.path.join(research_path, "debug_experiment", method, "single_triangle", "original", "bmp")

    for n, deg in enumerate(range(45, 46)):
        theta = np.radians(deg)
        R_p = h.Rotation_matrix(theta) # 回転行列
        G_total = np.zeros((Ny*2, Nx*2), dtype=np.complex128)
        face_indices = Global_faces
        rotated_vertices = Global_vertices @ R_p.T

        current_face_normals = Global_face_normals @ R_p.T                                       #三角形の法線ベクトル計算

        for i in range(Global_faces.shape[0]):
        # for i in range(1):
        # i=0
            n_vector = current_face_normals[i]
            current_face_indices = face_indices[i]
            current_face_vertices = rotated_vertices[current_face_indices]

            R = RotMatrix(n_vector)                                                 #回転行列計算
            Global_gravitypoints = np.mean(current_face_vertices, axis=0)           #グローバル座標系での三角形の重心計算
            local_vertices = (R @ (current_face_vertices - Global_gravitypoints).T).T     #ローカル座標系での三角形の頂点(z座標は0)
            
            """local座標系の(xhat, yhat)平面への変換"""
            local_vertices = local_vertices[:, :2]  #zhat成分は０なのでカット
            x_hat_min, y_hat_min = np.min(local_vertices, axis=0)
            x_hat_max, y_hat_max = np.max(local_vertices, axis=0)

            #ピクセル数の計算
            Nx_hat = int(np.ceil((x_hat_max - x_hat_min) / pitch))
            Ny_hat = int(np.ceil((y_hat_max - y_hat_min) / pitch))

            x_hat = np.linspace(x_hat_min, x_hat_max, Nx_hat)           #x_hat_minからx_hat_maxまでをNx_hatで分割して配列としてx_hatに格納
            y_hat = np.linspace(y_hat_max, y_hat_min, Ny_hat)           #y_hat_minからy_hat_maxまでをNy_hatで分割して配列としてy_hatに格納
            X_hat, Y_hat = np.meshgrid(x_hat, y_hat, indexing='ij')     #格子点の作成

            #三角形の内部判定
            points = np.vstack((X_hat.ravel(), Y_hat.ravel())).T
            path = mpltPath.Path(local_vertices)
            mask = path.contains_points(points).reshape(Nx_hat, Ny_hat)

            f = np.zeros((Nx_hat, Ny_hat), dtype=np.uint8)
            f[mask] = 1
            f = f.T
            # ゼロパディング
            f_pad = h.padding(f, Ny_hat, Nx_hat)

            #周波数座標系(u_hat, v_hat)の作成
            Nv_hat, Nu_hat = f_pad.shape
            u_hat = np.fft.fftshift(np.fft.fftfreq(Nu_hat, d=pitch))
            v_hat = -np.fft.fftshift(np.fft.fftfreq(Nv_hat, d=pitch))
            U_hat, V_hat = np.meshgrid(u_hat, v_hat)

            #ゼロパディング前のローカル座標系での三角形の重心計算
            Local_gravitypoints = np.mean(local_vertices, axis=0)
            x_hat_origin, y_hat_origin = Local_gravitypoints

            #ラスタライズしたときの原点がローカル座標系での座標値の計算
            O_x_hat = (x_hat_max - x_hat_min)/2
            O_y_hat = (y_hat_max - y_hat_min)/2

            x_hat_sft = O_x_hat - np.max(local_vertices[:, 0])
            y_hat_sft = O_y_hat - np.max(local_vertices[:, 1])

            # 位相シフト計算
            phase_shift = np.exp(1j * 2 * np.pi * (x_hat_sft*U_hat + y_hat_sft*V_hat))

            # 単一三角形のローカル座標系でのスペクトル計算
            F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f_pad)))
            F = h.FFT(f_pad)
            F = F * phase_shift

            Nv_hat_pad, Nu_hat_pad = F.shape
            u_hat_pad = np.fft.fftshift(np.fft.fftfreq(Nu_hat_pad, d=pitch))
            v_hat_pad = -np.fft.fftshift(np.fft.fftfreq(Nv_hat_pad, d=pitch))
            # U_hat_pad, V_hat_pad = np.meshgrid(u_hat_pad, v_hat_pad)

            # 補間法
            interp = RegularGridInterpolator(
                (v_hat_pad, u_hat_pad),
                F,
                method=method,
                bounds_error=False,
                fill_value=0
            )
            
            # interp_real = RectBivariateSpline(
            #     v_hat_pad[::-1], 
            #     u_hat_pad, 
            #     F[::-1, :].real, 
            #     kx=3, 
            #     ky=3
            # )

            # interp_imag = RectBivariateSpline(
            #     v_hat_pad[::-1], 
            #     u_hat_pad, 
            #     F[::-1, :].imag, 
            #     kx=3, 
            #     ky=3
            # )


            # x_interp_u = R[0, 0]*U + R[0, 1]*V + R[0, 2]*W - R[0, 2]/WAVELENGTH
            # y_interp_v = R[1, 0]*U + R[1, 1]*V + R[1, 2]*W - R[1, 2]/WAVELENGTH
            x_interp_u = R[0, 0]*U + R[0, 1]*V - R[0, 2]*WAVELENGTH*pre/2
            y_interp_v = R[1, 0]*U + R[1, 1]*V - R[1, 2]*WAVELENGTH*pre/2

            query_points = np.stack((y_interp_v.ravel(), x_interp_u.ravel()), axis=-1)
            # F_interp_real = interp_real(y_interp_v,x_interp_u, grid=False)
            # F_interp_imag = interp_imag(y_interp_v,x_interp_u, grid=False)
            F_interp = interp(query_points).reshape(Ny*2, Nx*2)
            # F_interp = (F_interp_real + 1j * F_interp_imag).reshape(Ny*2, Nx*2)

            #ヤコビアン行列計算
            Jr = R[0, 0]*R[1, 1] - R[0, 1]*R[1, 0]
            x_c, y_c, z_c = Global_gravitypoints
            # E1 = np.exp(-1j * 2 * np.pi * (U*x_c + V*y_c + W*z_c - z_c/WAVELENGTH))
            E1 = np.exp(-1j * 2 * np.pi * (U*x_c + V*y_c - z_c*WAVELENGTH*pre/2))

            # グローバル座標系での三角形のスペクトル
            G = Jr * E1 * F_interp

            G_total += G

        g_pad = h.IFFT(G_total)

        # 2048×2048を1024×1024に切り出し
        g = h.cutting(g_pad, Ny, Nx)

        g_view = h.normalize_255("log", g)

        holo_save_npy = os.path.join(model_npy, f"{n:03d}_{deg:+03d}.npy")
        holo_save_bmp = os.path.join(model_bmp, f"{n:03d}_{deg:+03d}.bmp")

        # np.save(holo_save_npy, g)
        Image.fromarray(g_view, mode="L").save(holo_save_bmp, format="BMP")