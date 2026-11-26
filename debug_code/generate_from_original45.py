"""Direct cubic transformations from Original +45, using free_rotation's mapping."""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
import hologram_utility as h


def main():
    root = Path(__file__).resolve().parent.parent
    base = root / 'debug_experiment/cubic/single_triangle'
    source = base / 'original/npy/090_+45.npy'
    output = base / 'direct_from_original_45_cubic'
    # Refuse to overwrite an earlier experiment.
    output.mkdir(exist_ok=False)
    for name in ('npy', 'bmp'):
        (output / name).mkdir()
    pitch, wavelength, z = 4.5e-3, 530e-6, 2.0
    g = np.load(source, allow_pickle=False)
    if g.shape != (512, 512) or not np.iscomplexobj(g) or not np.isfinite(g).all():
        raise ValueError('Expected a finite complex 512 x 512 hologram')
    ny, nx = g.shape
    u = np.fft.fftshift(np.fft.fftfreq(nx * 2, d=pitch))
    v = -np.fft.fftshift(np.fft.fftfreq(ny * 2, d=pitch))
    U, V = np.meshgrid(u, v)
    W = np.sqrt(wavelength**-2 - U**2 - V**2)
    forward = np.exp(2j * np.pi * z * W)
    spectrum = h.FFT(h.padding(g, ny, nx)) * forward
    interp = RegularGridInterpolator((v, u), spectrum, method='cubic',
                                     bounds_error=False, fill_value=0)
    rows = []
    for step in range(1, 46):
        delta = -step
        target = 45 + delta
        theta = np.deg2rad(delta)
        c, s = np.cos(theta), np.sin(theta)
        # R = [[c, 0, s], [0, 1, 0], [-s, 0, c]].
        up, vp = U - s / wavelength, V
        radicand = wavelength**-2 - up**2 - vp**2
        if np.any(radicand <= 0):
            raise ValueError(f'Non-propagating or singular coordinates at {delta}')
        wp = np.sqrt(radicand)
        alpha, beta = c * up + s * wp, vp
        # Correct derivative of wp; the y-axis determinant reduces to this.
        jacobian = np.abs(c - s * up / wp)
        rotated = interp(np.stack((beta, alpha), axis=-1)) * jacobian
        result = h.cutting(h.IFFT(rotated * forward.conj()), ny, nx)
        if not np.isfinite(result).all() or not np.any(result):
            raise ValueError(f'Invalid output at {delta}')
        stem = f'{step:03d}_delta{delta:+04d}_target{target:+04d}'
        np.save(output / 'npy' / f'{stem}.npy', result)
        Image.fromarray(h.normalize_255('log', result)).save(output / 'bmp' / f'{stem}.bmp')
        rows.append(dict(step=step, relative_angle_deg=delta,
                         nominal_target_angle_deg=target,
                         original_reference=f'original/npy/{target + 45:03d}_{target:+03d}.npy',
                         file=f'npy/{stem}.npy'))
        print(f'{step}/45: delta={delta:+d}, nominal target={target:+d}', flush=True)
    with (output / 'angles.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(source=str(source), interpolation='cubic', pitch=pitch,
                    wavelength=wavelength, propagation_before=z, propagation_after=-z,
                    input_angle_deg=45, relative_angles_deg=list(range(-1, -46, -1)),
                    mode='Each output independently uses Original +45; no chained transforms',
                    mapping='free_rotation_hologram.py with corrected Jacobian',
                    note='Target angles are nominal labels; rotation direction and agreement with Original remain unvalidated.')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(output, flush=True)


if __name__ == '__main__':
    main()
