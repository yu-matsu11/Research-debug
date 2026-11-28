"""Single-polygon comparison with Symeonidou et al., Eq. (14), axial pivot."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import hologram_utility as h
from PSNR import calculate_psnr, calculate_ssim


CASES = [(0, 0), (45, 45), (0, 1), (0, -1), (30, 29), (45, 44), (45, 42), (45, 40)]
PITCH, WAVELENGTH = 4.5e-3, 530e-6


def main():
    base = Path(__file__).resolve().parents[1] / "debug_experiment/cubic/single_triangle"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=base / "global_rotation_20261128")
    parser.add_argument("--pivot-z", type=float, default=2.0)
    args = parser.parse_args()
    z = args.pivot_z
    requested_output = args.output
    run = 1
    while True:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run += 1
            args.output = requested_output.with_name(f"{requested_output.name}_{run:02d}")
    print(f"Output: {args.output}", flush=True)

    def original(angle):
        return np.load(base / "original/npy" / f"{angle+45:03d}_{angle:+03d}.npy")

    ny, nx = original(0).shape
    u = np.fft.fftshift(np.fft.fftfreq(nx * 2, PITCH))
    v = -np.fft.fftshift(np.fft.fftfreq(ny * 2, PITCH))
    U, V = np.meshgrid(u, v)
    k = 1 / WAVELENGTH
    W = np.sqrt(k*k - U*U - V*V)
    H = np.exp(2j * np.pi * z * W)
    rows = []
    interpolators = {}
    for index, (start, target) in enumerate(CASES, 1):
        source, ref = original(start), original(target)
        padded = h.FFT(h.padding(source, ny, nx))
        if start not in interpolators:
            interpolators[start] = [
                RegularGridInterpolator((v, u), data, method="cubic",
                                        solver_args={"rtol": 1e-10, "atol": 0},
                                        bounds_error=False, fill_value=0)
                for data in (padded * H, padded)
            ]
        delta = target - start
        c, s = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))
        stem = f"{index:02d}_{start:+03d}_to_{target:+03d}"
        fields = [ref]
        for method, interp in zip(("carrier", "eq14"), interpolators[start]):
            X = U - k*s if method == "carrier" else U
            Z = np.sqrt(np.maximum(k*k - X*X - V*V, 0))
            alpha = c*X + s*Z
            gamma = -s*X + c*Z
            jac = np.abs(c - s*np.divide(X, Z, out=np.zeros_like(Z), where=Z>0))
            in_band = (alpha >= u.min()) & (alpha <= u.max())
            valid = in_band & (Z > 0) & (gamma > 0)
            mapped = np.zeros_like(padded)
            mapped[valid] = interp(np.column_stack((V[valid], alpha[valid]))) * jac[valid]
            if method == "eq14":
                # Evaluate propagation at the mapped physical frequency.
                mapped[valid] *= np.exp(2j * np.pi * z * gamma[valid])
            result = h.cutting(h.IFFT(mapped * H.conj()), ny, nx)
            if not np.isfinite(result).all():
                raise ValueError(f"Nonfinite output: {stem} {method}")
            np.save(args.output / f"{stem}_{method}.npy", result)
            fields.append(result)
            a, b = np.abs(ref), np.abs(result)
            span = np.ptp(a)
            a_norm, b_norm = (a-a.min())/span, (b-a.min())/span
            err = np.linalg.norm(result-ref) / np.linalg.norm(ref)
            rows.append(dict(case=index, input_deg=start, target_deg=target,
                             map_deg=delta, method=method, pivot_z_mm=z,
                             psnr_db=calculate_psnr(a_norm, b_norm),
                             ssim=calculate_ssim(a_norm, b_norm),
                             complex_relative_error=err,
                             out_of_band_fraction=1-in_band.mean(),
                             output_input_frobenius_ratio=np.linalg.norm(result)/np.linalg.norm(source),
                             output_input_energy_ratio=(np.linalg.norm(result)/np.linalg.norm(source))**2))
            if delta == 0 and err > 1e-8:
                raise RuntimeError(f"Zero rotation failed: {stem} {method}: {err}")
        fig, axes = plt.subplots(2, 3, figsize=(10, 7), layout="constrained")
        fft_ref = np.abs(h.FFT(ref)).max()
        reconstructed = [np.abs(h.cutting(h.IFFT(h.FFT(h.padding(g, ny, nx))*H), ny, nx))
                         for g in fields]
        for j, (title, field, image) in enumerate(zip(
                ("Original", "Carrier-shift baseline", "Eq. (14)"), fields, reconstructed)):
            fft_db = 20*np.log10(np.maximum(np.abs(h.FFT(field))/fft_ref, 1e-12))
            axes[0, j].imshow(fft_db, cmap="magma", vmin=-80, vmax=0)
            axes[0, j].set_title(title + " FFT")
            axes[1, j].imshow(image, cmap="gray", vmin=0, vmax=reconstructed[0].max())
            axes[1, j].set_title(f"Reconstruction at z={z:g} mm")
            for ax in axes[:, j]:
                ax.axis("off")
        fig.savefig(args.output / f"{stem}_comparison.png", dpi=160)
        plt.close(fig)
        print(stem, flush=True)
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
