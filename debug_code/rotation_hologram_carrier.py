"""キャリア付きホログラムの回転補償。依存ライブラリは numpy, scipy, matplotlib。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# パラメータ（普段はここを書き換える）
script_dir = Path(__file__).resolve().parent.parent
model_name = "single_triangle7"
dir_path = script_dir / "debug_npy" / f"{model_name}.npy"
if not dir_path.is_file():
    project_input = script_dir / "sources" / f"{model_name}.npy"
    if project_input.is_file():
        dir_path = project_input

pitch = 4.5e-3          # mm
WAVELENGTH = 530e-6      # mm
Nx, Ny = 512, 512
z = 2                  # 最初の伝搬距離 [mm]
z_first_focus = np.sqrt(3) / 2  # 1回目の回転後のフォーカス用伝搬距離 [mm]
z_second_focus = 0         # 2回目の回転後のフォーカス用伝搬距離 [mm]
theta_start = 0         # 初期角度 [deg]
theta_mid = 60          # 1回目の変換先 [deg]
theta_end = 0          # 2回目の変換先 [deg]。0 にすると 0 → 60 → 0
method = "cubic"


# コマンドラインで実行する場合は、必要な設定だけ上書きできる
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=dir_path)
parser.add_argument("--angles", type=float, nargs=3,
                    default=(theta_start, theta_mid, theta_end),
                    metavar=("START", "MID", "END"))
parser.add_argument("--round-trip", action="store_true")
parser.add_argument("--z", type=float, default=z)
parser.add_argument("--method", choices=("cubic", "linear"), default=method)
parser.add_argument("--jacobian", choices=("derived", "previous-code"), default="derived")
args = parser.parse_args()

theta_start, theta_mid, theta_end = args.angles
if args.round_trip:
    theta_end = theta_start
if not np.isfinite([theta_start, theta_mid, theta_end, args.z]).all():
    parser.error("角度と伝搬距離には有限値を指定してください。")
if not args.input.is_file():
    parser.error(f"入力が見つかりません。--input で指定してください。 {args.input}")
z = args.z


# グローバル座標系の周波数格子
k = 1.0 / WAVELENGTH
u = np.fft.fftshift(np.fft.fftfreq(Nx * 2, d=pitch))
v = -np.fft.fftshift(np.fft.fftfreq(Ny * 2, d=pitch))
U, V = np.meshgrid(u, v)
W = np.sqrt(np.maximum(k**2 - U**2 - V**2, 0.0))
X_start, Y_start = Nx // 2, Ny // 2


# 初期ホログラムを読み込み、z だけ伝搬させる
g = np.load(args.input, allow_pickle=False)
if g.shape != (Ny, Nx) or not np.issubdtype(g.dtype, np.number) or not np.isfinite(g).all():
    parser.error(f"入力には {Ny}×{Nx} の有限な実数または複素数の配列が必要です。")
g_pad = np.pad(g, ((Y_start, Y_start), (X_start, X_start)), mode="constant")
G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g_pad)))
Hz = np.exp(1j * 2 * np.pi * z * W)
G_recon = G * Hz
g_recon_pad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_recon)))
g_recon = g_recon_pad[Y_start:Y_start + Ny, X_start:X_start + Nx]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
g_view = np.log(np.abs(g_recon) + 1)
if g_view.max() > 0:
    g_view = 255 * (g_view / g_view.max())
axes[0].imshow(g_view, cmap="gray", vmin=0, vmax=255)
axes[0].set_title(f"Initial ({theta_start:g}°)")
axes[0].axis("off")


# ホログラム回転運動補償計算
# 1回目は初期角度 → 中間角度。2回目は中間角度 → 最終角度。
# どちらも同じ逆写像を使い、2回目には1回目の切り出し前のスペクトルを渡す。
G_input = G_recon
for ax, angle1, angle2, title, z_focus in (
    (axes[1], theta_start, theta_mid, "First rotation", z_first_focus),
    (axes[2], theta_mid, theta_end, "Second rotation", z_second_focus),
):
    # P1、P2 の絶対角度からそれぞれ回転行列を作る
    theta1 = np.radians(angle1)
    theta2 = np.radians(angle2)
    R1 = np.array([[np.cos(theta1), 0, np.sin(theta1)],
                   [0, 1, 0],
                   [-np.sin(theta1), 0, np.cos(theta1)]])
    R2 = np.array([[np.cos(theta2), 0, np.sin(theta2)],
                   [0, 1, 0],
                   [-np.sin(theta2), 0, np.cos(theta2)]])

    # 出力側 P2 の格子を P0 に戻す（列ベクトルに R2.T を掛ける）
    U2, V2, W2 = U, V, W
    A2 = R2[0, 0] * U2 + R2[1, 0] * V2 + R2[2, 0] * W2
    B2 = R2[0, 1] * U2 + R2[1, 1] * V2 + R2[2, 1] * W2
    C2 = R2[0, 2] * U2 + R2[1, 2] * V2 + R2[2, 2] * W2

    # P2 のキャリアを引き、P1 のキャリアを足して縦周波数を計算し直す
    A1 = A2 + (R1[2, 0] - R2[2, 0]) * k
    B1 = B2 + (R1[2, 1] - R2[2, 1]) * k
    C1 = np.sqrt(np.maximum(k**2 - A1**2 - B1**2, 0.0))

    # P0 → P1。入力スペクトルの参照位置を求める
    U1 = R1[0, 0] * A1 + R1[0, 1] * B1 + R1[0, 2] * C1
    V1 = R1[1, 0] * A1 + R1[1, 1] * B1 + R1[1, 2] * C1
    W1 = R1[2, 0] * A1 + R1[2, 1] * B1 + R1[2, 2] * C1

    # 前方伝搬枝を採用。接線付近、球面外、入力 FFT 格子外はゼロ。
    # 枝をまたぐ領域での可逆性は未検証。
    eps = k * 1e-12
    valid = (W2 > eps) & (C1 > eps) & (C2 > eps) & (W1 > eps)
    valid &= (U1 >= u.min() - eps) & (U1 <= u.max() + eps)
    valid &= (V1 >= v.min() - eps) & (V1 <= v.max() + eps)

    # 逆ヤコビアン。P2 → P0 が C2 / W2、P0 → P1 が W1 / C1。
    # 周波数座標の面積係数。光学的な振幅補正としての妥当性は未検証。
    numerator = np.abs(W1 * C2)
    denominator = np.abs(C1 * W2)
    if args.jacobian == "previous-code":
        numerator = np.abs(W1 * C1)
        denominator = np.abs(C2 * W2)
    jacobian = np.zeros_like(U2)
    np.divide(numerator, denominator, out=jacobian, where=valid)

    G_output = np.zeros_like(G_input)
    if np.any(valid):
        interp = RegularGridInterpolator((v, u), G_input, method=args.method)
        # 格子端に生じた丸め誤差を吸収する
        query_points = np.column_stack((
            np.clip(V1[valid], v.min(), v.max()),
            np.clip(U1[valid], u.min(), u.max()),
        ))
        G_output[valid] = interp(query_points) * jacobian[valid]
    print(f"{angle1:g}° → {angle2:g}°: 有効な出力周波数点 {valid.mean():.2%}")

    # 回転結果は次の回転と復元誤差の計算用に保持する
    g_output_pad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_output)))
    g_output = g_output_pad[Y_start:Y_start + Ny, X_start:X_start + Nx]

    # 表示用に指定距離の伝達関数を掛け、フォーカス面を再生する
    H_focus = np.exp(1j * 2 * np.pi * z_focus * W)
    g_focus_pad = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G_output * H_focus)))
    g_focus = g_focus_pad[Y_start:Y_start + Ny, X_start:X_start + Nx]
    g_view = np.log(np.abs(g_focus) + 1)
    if g_view.max() > 0:
        g_view = 255 * (g_view / g_view.max())
    ax.imshow(g_view, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{title} ({angle1:g}° → {angle2:g}°)\nz = {z_focus:.4g} mm")
    ax.axis("off")

    G_input = G_output

# 始点と終点の角度が同じ場合、正規化前の複素値で復元誤差を計算する
if np.isclose((theta_end - theta_start + 180) % 360 - 180, 0, atol=1e-10):
    for label, initial, restored in (
        ("複素スペクトル", G_recon, G_output),
        ("切り出し後の複素場", g_recon, g_output),
    ):
        initial_norm = np.linalg.norm(initial.ravel())
        if initial_norm > 0:
            error = np.linalg.norm((restored - initial).ravel()) / initial_norm
            print(f"{label}の相対 L2 誤差 {error:.6g}")
        else:
            print(f"{label}の相対 L2 誤差は算出不可（初期ノルムがゼロ）")

fig.tight_layout()
plt.show()
