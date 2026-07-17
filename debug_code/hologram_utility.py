import numpy as np

def FFT(X):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(X)))

def IFFT(X):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

def padding(X, Ny, Nx):
    return np.pad(X, ((Ny//2, Ny//2), (Nx//2, Nx//2)), 'constant')

def cutting(X, Ny, Nx):
    Y_start, X_start = Ny // 2, Nx // 2
    return X[Y_start : Y_start+Ny, X_start : X_start+Nx]

def normalize_255(mode, X):

    if mode == "abs":
        X_abs = np.abs(X)
        if X_abs.max() > 0:
            return (255 * (X_abs / X_abs.max())).astype(np.uint8)
        else:
            return np.zeros_like(X_abs, dtype=np.uint8)

    if mode == "angle":
        X_angle = np.angle(X)
        if X_angle.max() > 0:
            return (255 * (X_angle / X_angle.max())).astype(np.uint8)
        else:
            return np.zeros_like(X_angle, dtype=np.uint8)

    if mode == "log":
        X_log = np.log(np.abs(X)+1)
        if X_log.max() > 0:
            return (255 * (X_log / X_log.max())).astype(np.uint8)
        else:
            return np.zeros_like(X_log, dtype=np.uint8)

    if mode == "real":
        X_real = np.real(X)
        if X_real.max() > 0:
            return (255 * (X_real / X_real.max())).astype(np.uint8)
        else:
            return np.zeros_like(X_real, dtype=np.uint8)
        
def normalize_zero_to_one(mode, X):
    # 1. 各モードに応じてベースとなる実数データを抽出
    if mode == "abs":
        X_target = np.abs(X)
    elif mode == "angle":
        X_target = np.angle(X)
    elif mode == "log":
        X_target = np.log(np.abs(X) + 1.0)
    elif mode == "real":
        X_target = np.real(X)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2. 最小値と最大値を取得
    x_min = np.min(X_target)
    x_max = np.max(X_target)
    
    # 3. 最大値と最小値の差（データの幅）を計算
    x_range = x_max - x_min
    
    # 全て同じ値（差が0）の場合は、ゼロで埋めた配列を返す（ゼロ除算対策）
    if x_range == 0:
        return np.zeros_like(X_target, dtype=np.float64)
    
    # 4. 確実に 0.0 ~ 1.0 の範囲にスケーリング
    return (X_target - x_min) / x_range

def RotMatrix(n):
    n = n / np.linalg.norm(n)

    if abs(n[2] - 1.0) < 1e-9:
        return np.eye(3)

    ex, ez = [1, 0, 0], [0, 0, 1]
    cos_theta = ez @ n
    sin_theta = np.linalg.norm(np.cross(ez, n))
    nn = np.array([n[0], n[1], 0.0])
    nn_norm = np.linalg.norm(nn)
    if nn_norm == 0:
        return np.eye(3)
    nn = nn / nn_norm
    cos_phi = ex @ nn
    sin_phi = np.linalg.norm(np.cross(ex, nn))
    if n[1] < 0:
        sin_phi = -sin_phi

    R = np.array([[cos_phi*cos_theta, cos_theta*sin_phi, -sin_theta],
                  [-sin_phi,          cos_phi,            0],
                  [cos_phi*sin_theta, sin_phi*sin_theta,  cos_theta]])
    if abs(n[2] - 1) < 1e-6:
        R = np.eye(3)
    return R

def smp_WRP(V, tan_theta, pitch, Lx0, Ly0, dp2, d_WRP=0.0):
    """各三角形に必要なWRP上の小さい矩形領域を返す。

    V: shape (3, 3), グローバル座標の三角形頂点。行が x,y,z。
    """
    edge_f = np.abs(V[2, :] - d_WRP) * tan_theta
    box_lr = np.array([V[0, :] - edge_f, V[0, :] + edge_f])
    box_du = np.array([V[1, :] - edge_f, V[1, :] + edge_f])

    box_N = np.array([
        [np.floor(np.min(box_lr[0, :]) / pitch), np.ceil(np.max(box_lr[1, :]) / pitch)],
        [np.floor(np.min(box_du[0, :]) / pitch), np.ceil(np.max(box_du[1, :]) / pitch)]
    ], dtype=int)

    Nx_WRP = int(box_N[0, 1] - box_N[0, 0])
    Ny_WRP = int(box_N[1, 1] - box_N[1, 0])

    # FFTしやすいように偶数へ
    Nx_WRP += Nx_WRP % 2
    Ny_WRP += Ny_WRP % 2

    if Nx_WRP <= 2 or Ny_WRP <= 2:
        return None

    box_WRP_N = np.array([
        [box_N[0, 0], box_N[0, 0] + Nx_WRP],
        [box_N[1, 0], box_N[1, 0] + Ny_WRP]
    ], dtype=int)

    box_WRP = box_WRP_N.astype(float) * pitch

    # ホログラム範囲にクリップ
    x_min_bound, x_max_bound = -Lx0/2, Lx0/2 - pitch
    y_min_bound, y_max_bound = -Ly0/2 + pitch, Ly0/2

    box_WRP[0, 0] = max(box_WRP[0, 0], x_min_bound)
    box_WRP[0, 1] = min(box_WRP[0, 1], x_max_bound)
    box_WRP[1, 0] = max(box_WRP[1, 0], y_min_bound)
    box_WRP[1, 1] = min(box_WRP[1, 1], y_max_bound)

    Nx_WRP = int(round((box_WRP[0, 1] - box_WRP[0, 0]) / pitch))
    Ny_WRP = int(round((box_WRP[1, 1] - box_WRP[1, 0]) / pitch))
    Nx_WRP += Nx_WRP % 2
    Ny_WRP += Ny_WRP % 2

    if Nx_WRP <= 2 or Ny_WRP <= 2:
        return None

    Lx_WRP = Nx_WRP * pitch
    Ly_WRP = Ny_WRP * pitch

    fx_1d = np.linspace(-1/dp2, 1/dp2 - 1/Lx_WRP, Nx_WRP, endpoint=True)
    fy_1d = np.linspace( 1/dp2, -1/dp2 + 1/Ly_WRP, Ny_WRP, endpoint=True)
    fx, fy = np.meshgrid(fx_1d, fy_1d)

    sft_WRP = -np.mean(box_WRP, axis=1)
    return fx, fy, sft_WRP, box_WRP, Nx_WRP, Ny_WRP

def backface_culling(vertices, faces, face_normals, view_dir=np.array([0.0, 0.0, 1.0])):
    """頂点から求めた面法線により、view_dir を向く三角形だけを残す。

    比較元の ``backfaceCulling`` と同様に、法線と視線方向のなす角が
    88.9 度未満の面を表面として扱う。面積がゼロの三角形は除外する。
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    face_normals = np.asarray(face_normals)
    view_dir = np.asarray(view_dir, dtype=np.float64)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (N, 3)")
    if face_normals.shape[0] != faces.shape[0]:
        raise ValueError("faces and face_normals must have the same length")

    view_norm = np.linalg.norm(view_dir)
    if view_norm == 0:
        raise ValueError("view_dir must be a non-zero vector")
    view_dir = view_dir / view_norm

    triangle_vertices = vertices[faces.astype(np.intp, copy=False)]
    edge_ab = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge_ac = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    geometric_normals = np.cross(edge_ab, edge_ac)
    normal_lengths = np.linalg.norm(geometric_normals, axis=1)

    valid = normal_lengths > np.finfo(np.float64).eps
    unit_normals = np.zeros_like(geometric_normals)
    unit_normals[valid] = geometric_normals[valid] / normal_lengths[valid, None]

    threshold = np.cos(np.deg2rad(88.9))
    keep_mask = valid & ((unit_normals @ view_dir) >= threshold)
    # print(
    #     f"backface culling: before={faces.shape[0]}, "
    #     f"after={np.count_nonzero(keep_mask)}, "
    #     f"removed={faces.shape[0] - np.count_nonzero(keep_mask)}"
    # )
    return faces[keep_mask], face_normals[keep_mask], keep_mask

def ASM_prop_W(H, di, k, lam2, dp2, pitch, WAVELENGTH):
    """WRP面からホログラム面へ角スペクトル法で伝搬。"""
    if abs(di) < 1e-15:
        return H.copy()

    Ny0, Nx0 = H.shape
    H_pad = np.pad(H, ((Ny0//2, Ny0//2), (Nx0//2, Nx0//2)), 'constant')
    FH = FFT(H_pad)

    Nyp, Nxp = FH.shape
    Lx = Nxp * pitch
    Ly = Nyp * pitch

    fx_1d = np.linspace(-1/dp2, 1/dp2 - 1/Lx, Nxp, endpoint=True)
    fy_1d = np.linspace( 1/dp2, -1/dp2 + 1/Ly, Nyp, endpoint=True)
    fx, fy = np.meshgrid(fx_1d, fy_1d)

    # Band-limited ASM。diが小さい場合は極端に強い制限を避ける。
    fx_BL = (Nxp * pitch) / (lam2 * abs(di))
    fy_BL = (Nyp * pitch) / (lam2 * abs(di))
    mask = (np.abs(fx) <= fx_BL) & (np.abs(fy) <= fy_BL)

    root = 1.0 - (WAVELENGTH * fx)**2 - (WAVELENGTH * fy)**2
    root = np.maximum(root, 0.0)
    prop = np.exp(1j * k * di * np.sqrt(root)) * mask

    rH_pad = IFFT(FH * prop)
    return rH_pad[Nyp//2 - Ny0//2:Nyp//2 + Ny0//2,
                  Nxp//2 - Nx0//2:Nxp//2 + Nx0//2]

def fit_object_to_hologram(vertices_raw, Nx, Ny, pitch, fit_ratio=0.8):
    
    # ホログラム面の物理サイズ
    Lx0 = Nx * pitch
    Ly0 = Ny * pitch

    vertices_raw = vertices_raw.astype(np.float64)

    obj_x_size = vertices_raw[:, 0].max() - vertices_raw[:, 0].min()
    obj_y_size = vertices_raw[:, 1].max() - vertices_raw[:, 1].min()

    scale_x = (Lx0 * fit_ratio) / obj_x_size
    scale_y = (Ly0 * fit_ratio) / obj_y_size

    object_scale = min(scale_x, scale_y)

    Global_vertices = vertices_raw * object_scale

    # x,y中心をホログラム中心へ
    Global_vertices[:, 0] -= (
        Global_vertices[:, 0].max() + Global_vertices[:, 0].min()
    ) / 2

    Global_vertices[:, 1] -= (
        Global_vertices[:, 1].max() + Global_vertices[:, 1].min()
    ) / 2

    # z中心も0へ
    Global_vertices[:, 2] -= (
        Global_vertices[:, 2].max() + Global_vertices[:, 2].min()
    ) / 2

    return Global_vertices

def Rotation_matrix(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
              [0, 1, 0],
              [-np.sin(theta), 0, np.cos(theta)]])
    return R