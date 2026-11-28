from pathlib import Path
import csv, shutil, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd()
sys.path.insert(0,str(root/'debug_code'))
import hologram_utility as h
base=root/'debug_experiment/cubic/single_triangle'
src=base/'global_rotation_20261128'
out=root/'seminar/research_progress/2026-11-28'
figdir=out/'figures'
figdir.mkdir(parents=True,exist_ok=True)
shutil.copy2(src/'06_+45_to_+44_comparison.png',figdir/'comparison.png')
shutil.copy2(src/'metrics.csv',out/'metrics.csv')
rows=list(csv.DictReader((src/'metrics.csv').open()))
fig,axes=plt.subplots(3,2,figsize=(6.5,8.3),layout='constrained')
overlay=[]
def reconstruct(g):
    ny,nx=g.shape
    u=np.fft.fftfreq(nx*2,0.0045)
    v=np.fft.fftfreq(ny*2,0.0045)
    W=np.fft.fftshift(np.sqrt(0.00053**-2-u[None,:]**2-v[:,None]**2))
    return np.abs(h.cutting(h.IFFT(h.FFT(h.padding(g,ny,nx))*np.exp(4j*np.pi*W)),ny,nx))
for i,case in enumerate((4,5,6)):
    pair=[r for r in rows if int(r['case'])==case]
    a,b=(int(pair[0][k]) for k in ('input_deg','target_deg'))
    ref=reconstruct(np.load(base/'original/npy'/f'{b+45:03d}_{b:+03d}.npy'))
    threshold=0.5*ref.max()
    mask=ref>=threshold
    for j,method in enumerate(('carrier','eq14')):
        test=reconstruct(np.load(src/f'{case:02d}_{a:+03d}_to_{b:+03d}_{method}.npy'))
        m=test>=threshold
        ax=axes[i,j]
        ax.imshow(ref,cmap='gray',vmin=0,vmax=ref.max())
        ax.contour(mask,levels=[.5],colors=['cyan'],linewidths=.75)
        if m.any():
            ax.contour(m,levels=[.5],colors=['orange'],linewidths=.75)
        iou=(mask&m).sum()/(mask|m).sum()
        shift=np.linalg.norm(np.argwhere(mask).mean(axis=0)-np.argwhere(m).mean(axis=0)) if m.any() else float('nan')
        overlay.append(dict(case=case,method=method,iou=iou,centroid_distance_px=shift))
        ax.set_title(f'{a} to {b} deg | {method} | IoU {iou:.3f}',fontsize=9)
        ax.axis('off')
fig.savefig(figdir/'overlay.png',dpi=190)
with (out/'overlay_metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(overlay[0]))
    writer.writeheader()
    writer.writerows(overlay)
print(overlay)
for i in range(0,len(rows),2):
    a,b=rows[i:i+2]
    print(a['input_deg'],a['target_deg'],*[round(float(r[k]),4) for r in (a,b) for k in ('psnr_db','ssim','complex_relative_error','output_input_frobenius_ratio','out_of_band_fraction')])
