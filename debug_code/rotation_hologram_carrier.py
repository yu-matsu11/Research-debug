"""キャリア付きホログラムの回転補償。依存ライブラリは numpy, scipy, Pillow。

角度は Y 軸まわりの絶対角度（度）。既定値は 0 → 60 → 30。
--round-trip では始点の角度に戻す（既定の始点・中間角では 0 → 60 → 0）。

逆写像は P2 → P0 → P1 の両中間ベクトルで前方伝搬枝を採用する。
負の枝、接線付近、球面外、入力 FFT 格子外はゼロとする。
J は周波数座標の変数変換に対する面積係数。物理的な振幅補正としての
妥当性（共通中間面でのスペクトルの一致など）は別途検証が必要。
前回答のサンプルコードでは C1 と C2 の位置が導出式と逆だった。
既定値は導出式の w1*C2/(C1*w2)。--jacobian previous-code で当時の係数を再現する。
補間と帯域の打ち切りにより、往復しても離散複素場は一般に一致しない。
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator


PITCH = 4.5e-3          # mm
WAVELENGTH = 530e-6      # mm
NX, NY = 512, 512


def rotation_matrix(theta):
    """ラジアン指定。P0 から各ローカル座標系への変換に使う。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def inverse_frequency_map(u2, v2, w2, r1, r2, k, jacobian_mode="derived"):
    """P2 の格子から P1 の補間座標、逆ヤコビアン、有効マスクを返す。"""
    q2 = np.stack((u2, v2, w2), axis=-1)
    # 行ベクトル表記。列ベクトルでは R2.T @ q2 に相当する。
    q02 = q2 @ r2
    # P2 のキャリアを除去し、P1 のキャリアを加えて縦成分を再計算。
    a1, b1 = np.moveaxis(q02[..., :2] + k * (r1[2, :2] - r2[2, :2]), -1, 0)
    c1 = np.sqrt(np.maximum(k**2 - a1**2 - b1**2, 0.0))
    q1 = np.stack((a1, b1, c1), axis=-1) @ r1.T
    u1, v1, w1 = np.moveaxis(q1, -1, 0)
    c2 = q02[..., 2]
    eps = k * 1e-12
    valid = (w2 > eps) & (c1 > eps) & (c2 > eps) & (w1 > eps)
    jacobian = np.zeros_like(w2)
    # P2 → P0 の面積比は C2/W2、P0 → P1 の面積比は W1/C1。
    if jacobian_mode == "previous-code":
        c1, c2 = c2, c1
    np.divide(np.abs(w1 * c2), np.abs(c1 * w2), out=jacobian, where=valid)
    return u1, v1, jacobian, valid


def rotate_spectrum(spectrum, u, v, grid, angle_from, angle_to, method, jacobian_mode):
    k = 1.0 / WAVELENGTH
    r1 = rotation_matrix(np.deg2rad(angle_from))
    r2 = rotation_matrix(np.deg2rad(angle_to))
    u1, v1, jacobian, valid = inverse_frequency_map(*grid, r1, r2, k, jacobian_mode)
    # 格子端の丸め誤差だけ許容する。実際に帯域外に出た点はゼロ。
    tolerance = k * 1e-12
    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()
    valid &= (u1 >= u_min - tolerance) & (u1 <= u_max + tolerance)
    valid &= (v1 >= v_min - tolerance) & (v1 <= v_max + tolerance)
    output = np.zeros_like(spectrum)
    if np.any(valid):
        interp = RegularGridInterpolator((v, u), spectrum, method=method)
        points = np.column_stack((
            np.clip(v1[valid], v_min, v_max),
            np.clip(u1[valid], u_min, u_max),
        ))
        output[valid] = interp(points) * jacobian[valid]
    print(f"{angle_from:g}° → {angle_to:g}°: 有効な出力周波数点 {valid.mean():.2%}", flush=True)
    return output


def image_field(spectrum):
    padded = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    return padded[NY // 2:NY // 2 + NY, NX // 2:NX // 2 + NX]


def save_image(field, path):
    # 添付の normalize_255("log", ...) と同じ各画像ごとの正規化。
    magnitude = np.log(np.abs(field) + 1)
    peak = magnitude.max()
    if peak > 0:
        magnitude = 255 * (magnitude / peak)
    Image.fromarray(magnitude.astype(np.uint8)).save(path)


def relative_error(reference, result):
    norm = np.linalg.norm(reference.ravel())
    if norm == 0:
        return "算出不可（初期ノルムがゼロ）"
    return f"{np.linalg.norm((result - reference).ravel()) / norm:.6g}"


def main():
    project_dir = Path(__file__).resolve().parent.parent
    original_input = project_dir / "debug_npy" / "single_triangle7.npy"
    mirrored_input = project_dir / "sources" / "single_triangle7.npy"
    default_input = mirrored_input if not original_input.is_file() and mirrored_input.is_file() else original_input

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=default_input, help="512×512 の入力 .npy")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "seminar", help="BMP の保存先")
    parser.add_argument("--angles", type=float, nargs=3, default=(0, 60, 30),
                        metavar=("START", "MID", "END"), help="絶対角度（度）、既定値 0 60 30")
    parser.add_argument("--round-trip", action="store_true", help="終点を始点と同じ角度にする")
    parser.add_argument("--z", type=float, default=2, help="最初の伝搬距離（mm）、既定値 2")
    parser.add_argument("--method", choices=("cubic", "linear"), default="cubic", help="補間方法")
    parser.add_argument("--jacobian", choices=("derived", "previous-code"), default="derived",
                        help="導出式の係数、または前回答のサンプルコードの係数")
    args = parser.parse_args()
    start, mid, end = args.angles
    if args.round_trip:
        end = start
    if not np.isfinite([start, mid, end, args.z]).all():
        parser.error("角度と伝搬距離には有限値を指定してください。")
    if not args.input.is_file():
        parser.error(f"入力が見つかりません。--input で指定してください。 {args.input}")
    field = np.load(args.input, allow_pickle=False)
    if field.shape != (NY, NX) or not np.issubdtype(field.dtype, np.number) or not np.isfinite(field).all():
        parser.error(f"入力には {NY}×{NX} の有限な実数または複素数の配列が必要です。")

    u = np.fft.fftshift(np.fft.fftfreq(2 * NX, d=PITCH))
    v = -np.fft.fftshift(np.fft.fftfreq(2 * NY, d=PITCH))
    uu, vv = np.meshgrid(u, v)
    w = np.sqrt(WAVELENGTH**-2 - uu**2 - vv**2)
    grid = uu, vv, w

    padded = np.pad(field, ((NY // 2, NY // 2), (NX // 2, NX // 2)))
    initial = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    initial *= np.exp(2j * np.pi * args.z * w)
    rotated = rotate_spectrum(initial, u, v, grid, start, mid, args.method, args.jacobian)
    restored = rotate_spectrum(rotated, u, v, grid, mid, end, args.method, args.jacobian)

    # 伝搬は冒頭で一度だけ。2回目は切り出し前のスペクトルを直接使う。
    initial_field = image_field(initial)
    restored_field = image_field(restored)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_image(initial_field, args.output_dir / "initial_hologram.bmp")
    save_image(image_field(rotated), args.output_dir / "rotated_hologram.bmp")
    save_image(restored_field, args.output_dir / "restored_hologram.bmp")
    print(f"保存先 {args.output_dir.resolve()}")

    if np.isclose((end - start + 180) % 360 - 180, 0, atol=1e-10):
        print(f"複素スペクトルの相対 L2 誤差 {relative_error(initial, restored)}")
        print(f"切り出し後の複素場の相対 L2 誤差 {relative_error(initial_field, restored_field)}")


if __name__ == "__main__":
    main()
