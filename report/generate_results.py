"""Reproduce the report's measurements, plots, and LaTeX result tables.

Run from the repository root: python -m report.generate_results
Use --reuse to reuse output/report-data from generate_assets.py.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from attacks.compression import jpeg_compress
from attacks.spatial import awgn
from benchmarks.run_benchmarks import default_attacks, run_image_attack_suite
from core import ImageWatermarkConfig, ImageWatermarker, ImageWatermarkExtractor
from core.metrics import byte_bit_error_rate, psnr, ssim
from generate_assets import DEFAULT_KEY, DEFAULT_MESSAGE, generate_video_benchmark_assets, _save_comparison, _save_residual, _save_dwt_subbands, _save_compression_curve

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
DATA = REPORT / "data"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed in this run (video results reused)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    payload = DEFAULT_MESSAGE.encode()
    config = ImageWatermarkConfig()
    cover = cv2.imread(str(FIGURES / "nature_original.png"))
    photo = cv2.imread(str(FIGURES / "rocket_original.png"))
    if cover is None or photo is None:
        raise FileNotFoundError("The supplied nature image and rocket photograph fixtures are required")
    marked, _ = ImageWatermarker(config).embed_array(cover, payload, DEFAULT_KEY, utf8=True)
    rows = [r.to_dict() for r in run_image_attack_suite(cover, marked, payload, DEFAULT_KEY, config)]
    save_json(DATA / "robustness.json", rows)
    _save_comparison(cover, marked, FIGURES / "original_vs_watermarked.png", psnr_db=psnr(cover, marked), ssim_value=ssim(cover, marked))
    _save_residual(cover, marked, FIGURES / "residual_x10.png")
    _save_dwt_subbands(cover, marked, config, FIGURES / "dwt_subbands.png")
    curve = _save_compression_curve(cover, marked, payload, DEFAULT_KEY, config, FIGURES / "compression_curve_unused.png")
    (FIGURES / "compression_curve_unused.png").unlink()
    save_json(DATA / "compression_curve.json", curve)
    photo_marked, _ = ImageWatermarker(config).embed_array(photo, payload, DEFAULT_KEY, utf8=True)
    photo_rows = [r.to_dict() for r in run_image_attack_suite(photo, photo_marked, payload, DEFAULT_KEY, config)]
    save_json(DATA / "rocket_results.json", {"fixture": "SpaceX rocket photograph via scikit-image data, resized to 512x512", "payload": DEFAULT_MESSAGE, "rows": photo_rows})
    save_json(DATA / "summary.json", {"fixture": "AI-generated forest and river landscape, resized to 512x512", "message": DEFAULT_MESSAGE,
        "config": config.to_dict(), "baseline": {"psnr_db": psnr(cover, marked), "ssim": ssim(cover, marked), "authenticated": rows[0]["valid"]}, "benchmark": rows})
    if not args.reuse or not (DATA / "video_robustness.json").exists():
        generate_video_benchmark_assets(DATA)
    video = json.loads((DATA / "video_robustness.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    for ax, image, title in zip(axes, [photo, photo_marked], ["Original rocket photograph", "Watermarked rocket photograph"]):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(FIGURES / "rocket_comparison.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.7), layout="constrained")
    for ax, (name, attack, geometric), row in zip(axes.flat, default_attacks(), rows):
        ax.imshow(cv2.cvtColor(attack(marked), cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}\nBER {row['ber']:.4f} | {'Pass' if row['valid'] else 'Fail'}", fontsize=10)
        ax.axis("off")
    fig.savefig(FIGURES / "attack_montage.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
    for ax, field, label, color in zip(axes, ["psnr_db", "ssim", "ber"],
            ["PSNR (dB)", "SSIM", "Payload BER"], ["#246a91", "#278477", "#bd5538"]):
        ax.plot([r["quality"] for r in curve], [r[field] for r in curve], "o-", color=color)
        ax.set(xlabel="JPEG quality factor", ylabel=label, xlim=(100, 30))
        ax.grid(alpha=.25)
        if field == "ber":
            ax.set_ylim(-.02, 1.02)
    fig.savefig(FIGURES / "jpeg_sweep.png", dpi=220)
    plt.close(fig)

    sweep = []
    for delta in [4, 6, 8, 10, 12, 16, 20]:
        config = ImageWatermarkConfig(delta=float(delta))
        image, _ = ImageWatermarker(config).embed_array(cover, payload, DEFAULT_KEY, utf8=True)
        row = {"delta": delta, "psnr_db": psnr(cover, image), "ssim": ssim(cover, image)}
        extractor = ImageWatermarkExtractor(config)
        for name, attacked in [("clean", image), ("jpeg50", jpeg_compress(image, 50)),
                               ("awgn10", awgn(image, 10, seed=7))]:
            result = extractor.extract_array(attacked, DEFAULT_KEY, expected_payload_bytes=len(payload))
            row[name + "_ber"] = byte_bit_error_rate(payload, result.payload)
            row[name + "_valid"] = result.valid
        sweep.append(row)
    save_json(DATA / "strength_sweep.json", sweep)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    axes[0].plot([r['delta'] for r in sweep], [r['psnr_db'] for r in sweep], 'o-', color='#246a91')
    axes[0].set(xlabel=r'Quantization step $\Delta$', ylabel='Unattacked PSNR (dB)')
    for field, label in [('jpeg50_ber', 'JPEG Q=50'), ('awgn10_ber', 'AWGN sigma=10')]:
        axes[1].plot([r['delta'] for r in sweep], [r[field] for r in sweep], 'o-', label=label)
    axes[1].set(xlabel=r'Quantization step $\Delta$', ylabel='Payload BER', ylim=(-.02, 1.02))
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(FIGURES / 'strength_sweep.png', dpi=220)
    plt.close(fig)

    # A result-independent schematic; all other plots use measured values.
    fig, ax = plt.subplots(figsize=(11, 3.5), layout='constrained')
    ax.axis('off')
    labels = ['Brightness Y', '3-level\nHaar DWT', '4 x 4\nblock DCT', 'Keyed STDM', 'Inverse\ntransforms']
    xs = np.linspace(.09, .91, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        ax.text(x, .62, label,
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.7', fc='#e8f1f6', ec='#246a91'), fontsize=10)
        if i:
            ax.annotate('', (x-.08, .62), (xs[i-1]+.08, .62), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='#246a91', lw=1.5))
    ax.text(.5, .18, 'Message -> framing + whitening + convolutional code -> bits\nSecret key -> carrier order + PN chips + dither',
            ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.annotate('', (.705,.48), (.60,.28), xycoords='axes fraction', arrowprops=dict(arrowstyle='->'))
    fig.savefig(FIGURES / 'pipeline.png', dpi=220)
    plt.close(fig)

    names = ['No attack', 'JPEG Q=50', r'Rotation $5^\circ$ (registered)',
             r'Median $3\times3$', r'AWGN $\sigma=10$', r'Gaussian blur $3\times3$',
             r'Scale $0.75\times$ (registered)', r'Crop 90\% per axis (registered)']
    table = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Attack & PSNR (dB) & SSIM & BER & Valid \\', r'\midrule']
    for name, row in zip(names, rows):
        table.append(f"{name} & {row['psnr_db']:.2f} & {row['ssim']:.4f} & {row['ber']:.4f} & {'Yes' if row['valid'] else 'No'} " + r'\\')
    table += [r'\bottomrule', r'\end{tabular}']
    (REPORT / 'image_results.tex').write_text('\n'.join(table)+'\n', encoding='utf-8')
    table = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Attack & PSNR (dB) & SSIM & BER & Valid \\', r'\midrule']
    for name, row in zip(names, photo_rows):
        table.append(f"{name} & {row['psnr_db']:.2f} & {row['ssim']:.4f} & {row['ber']:.4f} & {'Yes' if row['valid'] else 'No'} " + r'\\')
    table += [r'\bottomrule', r'\end{tabular}']
    (REPORT / 'rocket_results.tex').write_text('\n'.join(table)+'\n', encoding='utf-8')
    table = [r'\begin{tabular}{lrrr}', r'\toprule', r'Video condition & BER & Valid & Samples \\', r'\midrule']
    for row in video['rows']:
        table.append(f"{row['attack']} & {row['ber']:.4f} & {'Yes' if row['valid'] else 'No'} & {row['temporal_samples']} " + r'\\')
    table += [r'\bottomrule', r'\end{tabular}']
    (REPORT / 'video_results.tex').write_text('\n'.join(table)+'\n', encoding='utf-8')
    table = [r'\begin{tabular}{rrrrr}', r'\toprule',
             r'$\Delta$ & Clean PSNR (dB) & Clean SSIM & JPEG BER & AWGN BER \\', r'\midrule']
    for row in sweep:
        table.append(f"{row['delta']} & {row['psnr_db']:.2f} & {row['ssim']:.4f} & {row['jpeg50_ber']:.4f} & {row['awgn10_ber']:.4f} " + r'\\')
    table += [r'\bottomrule', r'\end{tabular}']
    (REPORT / 'strength_results.tex').write_text('\n'.join(table)+'\n', encoding='utf-8')
    save_json(DATA / 'environment.json', {
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
        'python': platform.python_version(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'packages': {n: package_version(n) for n in ['numpy','scipy','PyWavelets','opencv-python','matplotlib','ffmpeg-python']},
        'ffmpeg': subprocess.check_output(['ffmpeg','-version'], text=True).splitlines()[0],
        'image_sources': ['AI-generated forest and river landscape resized to 512x512','SpaceX rocket photograph resized to 512x512'], 'awgn_seed': 7, 'image_payload': DEFAULT_MESSAGE,
        'image_key': DEFAULT_KEY, 'note': 'Public test key; one AI-generated image, one photographic image, and a synthetic video. This small set is not representative of AI-generated media.'})
    print('Report figures, tables, and data written to', REPORT)
    print('Strength sweep:', json.dumps(sweep))


if __name__ == '__main__':
    main()
