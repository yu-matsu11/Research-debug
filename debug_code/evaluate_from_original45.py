"""Evaluate the direct +45 experiment using the existing PSNR.py metrics."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PSNR import calculate_psnr, calculate_ssim


def main():
    base = Path(__file__).resolve().parent.parent / 'debug_experiment/cubic/single_triangle'
    experiment = base / 'direct_from_original_45_cubic'
    output = experiment / 'evaluation'
    output.mkdir(exist_ok=True)
    with (experiment / 'angles.csv').open(encoding='utf-8-sig', newline='') as f:
        pairs = list(csv.DictReader(f))
    results = []
    for pair in pairs:
        ref_path = base / pair['original_reference']
        test_path = experiment / pair['file']
        ref = np.abs(np.load(ref_path, allow_pickle=False)).astype(np.float64)
        test = np.abs(np.load(test_path, allow_pickle=False)).astype(np.float64)
        if ref.shape != test.shape or not np.isfinite(ref).all() or not np.isfinite(test).all():
            raise ValueError(f'Invalid pair: {test_path}')
        low, span = ref.min(), np.ptp(ref)
        if span == 0:
            raise ValueError(f'Zero reference range: {ref_path}')
        # Same amplitude normalization as PSNR.process_method; no clipping.
        ref_norm, test_norm = (ref - low) / span, (test - low) / span
        results.append(dict(relative_angle_deg=int(pair['relative_angle_deg']),
                            target_angle_deg=int(pair['nominal_target_angle_deg']),
                            psnr_db=float(calculate_psnr(ref_norm, test_norm)),
                            ssim=float(calculate_ssim(ref_norm, test_norm)),
                            original_file=str(ref_path.relative_to(base)),
                            generated_file=pair['file']))
    with (output / 'psnr_ssim_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for key, ylabel, filename in [('psnr_db', 'PSNR (dB)', 'psnr.png'),
                                  ('ssim', 'SSIM', 'ssim.png')]:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot([r['target_angle_deg'] for r in results], [r[key] for r in results],
                'o-', markersize=3, linewidth=1, label='Cubic: fixed Original +45 input')
        ax.set(xlabel='Nominal output angle (degrees)', ylabel=ylabel,
               title=f'{ylabel}: transformed hologram vs Original', xlim=(45, 0))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(output / filename, dpi=200)
        plt.close(fig)
    summary = {'count': len(results), 'domain': 'Hologram amplitude, not reconstructed image or phase',
               'normalization': '(abs(data) - Original.min) / (Original.max - Original.min); data_range=1; no clipping',
               'metric_implementation': 'debug_code/PSNR.py calculate_psnr and calculate_ssim',
               'angle_labels': 'Nominal target = 45 + relative angle; direction not independently validated'}
    for key in ('psnr_db', 'ssim'):
        values = np.array([r[key] for r in results])
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite {key}')
        summary[key] = dict(mean=float(values.mean()), min=float(values.min()), max=float(values.max()))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    for angle in (44, 30, 0):
        print(next(r for r in results if r['target_angle_deg'] == angle))


if __name__ == '__main__':
    main()
