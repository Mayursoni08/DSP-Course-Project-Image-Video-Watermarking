# Benchmarking protocol

## Reproducible fixture

`benchmarks.sample_data.synthetic_cover()` creates a deterministic 512×512 BGR image containing gradients, correlated texture, smooth regions, sharp edges, repeated line features, text, and geometric shapes. No downloaded or restricted asset is required.

The committed benchmark uses:

```text
key:     docs-only-demo-key-2026
payload: DSP-WATERMARK-2026 (18 UTF-8 bytes)
config:  default ImageWatermarkConfig
seed:    2026
```

The key is intentionally public because this is a repeatable test vector, not a production secret.

## Image metrics

- **PSNR:** $10\log_{10}(255^2/\mathrm{MSE})$ over all pixels/channels.
- **SSIM:** Gaussian-window SSIM with 11×11 window and $\sigma=1.5$, averaged over channels.
- **BER:** unequal payload bits divided by the larger expected/extracted bit count; missing bits count as errors.
- **NC:** zero-mean normalized correlation for numeric watermark arrays.

No-attack PSNR and SSIM quantify embedding imperceptibility. Attacked PSNR and SSIM quantify the complete distortion channel, not only the watermark.

## Image attacks

The default suite applies:

```text
No attack
JPEG quality 50
Rotation 5 degrees + reference registration
Median filter 3x3
AWGN sigma 10, deterministic seed 7
Gaussian blur 3x3, sigma 0.8
Scale 0.75x + reference registration
Center crop 90% per axis + resize + reference registration
```

Run:

```bash
python generate_assets.py
# or
python -m benchmarks.run_benchmarks --input assets/original.png \
  --key docs-only-demo-key-2026 --message DSP-WATERMARK-2026
```

## Video protocol

Create or supply a video of at least 24 frames at 512×512, embed the payload once, then test:

```text
No additional attack
H.264 CRF 28 recompression
H.265 CRF 30 recompression
Drop every 8th frame
Frame-rate conversion to 75% of source FPS
```

Run:

```bash
python -m benchmarks.run_video_benchmarks \
  --input input.mp4 \
  --key a-long-test-key \
  --message VIDEO \
  --output-dir results/video
```

Video results include BER, frame-phase selected by the decoder, number of temporal score samples, and authentication state. A larger and more diverse dataset is required before claiming population-level performance.

## Reporting policy

Commit raw JSON/CSV beside any table or graph. Record codec version, platform, content class, dimensions, payload length, and complete config for formal experiments. Report failures and confidence intervals; do not select only successful covers.
