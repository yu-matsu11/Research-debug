import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(csv_path):
    angles = []
    psnr_values = []
    ssim_values = []

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            angles.append(float(row["Angle"]))
            psnr_values.append(float(row["PSNR (dB)"]))
            ssim_values.append(float(row["SSIM"]))

    return angles, psnr_values, ssim_values


def main():
    experiment_dir = Path(__file__).resolve().parent.parent / "debug_experiment"
    methods = {
        "Linear": experiment_dir / "linear" / "psnr_ssim_linear_results.csv",
        "Cubic": experiment_dir / "cubic" / "psnr_ssim_cubic_results.csv",
    }
    colors = {"Linear": "tab:blue", "Cubic": "tab:orange"}

    psnr_fig, psnr_ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ssim_fig, ssim_ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for method, csv_path in methods.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        angles, psnr_values, ssim_values = load_results(csv_path)
        psnr_ax.scatter(
            angles, psnr_values, s=24, alpha=0.8, color=colors[method], label=method
        )
        ssim_ax.scatter(
            angles, ssim_values, s=24, alpha=0.8, color=colors[method], label=method
        )

    psnr_ax.set_title("PSNR")
    psnr_ax.set_xlabel("Angle (degrees)")
    psnr_ax.set_ylabel("PSNR (dB)")
    psnr_ax.grid(True, alpha=0.3)
    psnr_ax.legend()

    ssim_ax.set_title("SSIM")
    ssim_ax.set_xlabel("Angle (degrees)")
    ssim_ax.set_ylabel("SSIM")
    ssim_ax.grid(True, alpha=0.3)
    ssim_ax.legend()

    psnr_output_path = experiment_dir / "psnr_scatter.png"
    ssim_output_path = experiment_dir / "ssim_scatter.png"
    psnr_fig.savefig(psnr_output_path, dpi=300)
    ssim_fig.savefig(ssim_output_path, dpi=300)
    print(f"PSNR graph saved: {psnr_output_path}")
    print(f"SSIM graph saved: {ssim_output_path}")
    plt.show()


if __name__ == "__main__":
    main()
