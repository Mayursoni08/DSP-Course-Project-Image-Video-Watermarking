# DSP Theory and Implementation

This document preserves the mathematical and algorithmic explanations from the original project README for the Digital Signal Processing course project. For installation and commands, see the [main README](../README.md).

The sections below explain the transforms, keyed embedding rule, error correction, reconstruction, and temporal processing implemented in Python. Configuration tables record the default operating parameters.

## Where the mathematics appears in Python

| Processing stage | Implementation |
|---|---|
| Color conversion and luminance handling | [`core/media.py`](../core/media.py) |
| Padding, spatial/temporal DWT, block DCT, and SVD utilities | [`core/transforms.py`](../core/transforms.py) |
| Sub-band selection and carrier layout | [`core/carriers.py`](../core/carriers.py) |
| Keyed PN projections and STDM embedding/extraction | [`core/spread_spectrum.py`](../core/spread_spectrum.py) |
| Dithered parity-QIM decisions | [`core/qim.py`](../core/qim.py) |
| Framing, whitening, CRC/HMAC, convolutional coding, and Viterbi decoding | [`core/payload.py`](../core/payload.py) |
| Image/video embedding and reconstruction | [`core/embedder.py`](../core/embedder.py) |
| Extraction, temporal voting, and phase search | [`core/extractor.py`](../core/extractor.py) |
| Geometric alignment | [`core/registration.py`](../core/registration.py) |
| PSNR, SSIM, BER, and normalized correlation | [`core/metrics.py`](../core/metrics.py) |
| Configuration and validation | [`core/config.py`](../core/config.py) |

For a worked carrier-capacity calculation and implementation conventions, see the [algorithm reference](ALGORITHM.md). For experimental methods and metric definitions, see the [benchmarking protocol](BENCHMARKING.md).

## DSP architecture

### 1. Luminance-domain processing

For color media, the input BGR frame is converted to YCrCb and only the luminance plane is modified. Chroma and any alpha channel are preserved. This concentrates the watermark in a channel that survives common codecs while avoiding visible color shifts.

Let the host luminance plane be

$$
Y \in [0,255]^{H\times W}.
$$

It is reflect-padded to a multiple of

$$
M = 2^L B,
$$

where $L$ is the DWT level and $B$ is the DCT block size. Padding metadata is retained and removed after reconstruction.

### 2. Multi-level 2D-DWT decomposition

A separable wavelet decomposition applies low-pass $h[n]$ and high-pass $g[n]$ analysis filters along rows and columns, followed by dyadic downsampling. At one level,

$$
\begin{aligned}
LL(i,j) &= \sum_m\sum_n Y(m,n)h(2i-m)h(2j-n),\\
LH(i,j) &= \sum_m\sum_n Y(m,n)h(2i-m)g(2j-n),\\
HL(i,j) &= \sum_m\sum_n Y(m,n)g(2i-m)h(2j-n),\\
HH(i,j) &= \sum_m\sum_n Y(m,n)g(2i-m)g(2j-n).
\end{aligned}
$$

The approximation $LL$ is recursively decomposed to level $L=3$. The default embeds into `LH₃` and `HL₃`: these mid-frequency orientation bands offer a better imperceptibility/robustness compromise than altering the low-frequency approximation or the most fragile diagonal detail band.

Run `python generate_assets.py` from the repository root to generate `assets/dwt_subbands.png`, which visualizes the wavelet sub-bands and embedding support.

### 3. Block DCT and coefficient selection

Each selected DWT band is partitioned into $B\times B$ blocks. For a block $x[m,n]$, the orthonormal 2D DCT-II is

$$
C[u,v] = \alpha(u)\alpha(v)
\sum_{m=0}^{B-1}\sum_{n=0}^{B-1}
x[m,n]
\cos\!\left[\frac{\pi(2m+1)u}{2B}\right]
\cos\!\left[\frac{\pi(2n+1)v}{2B}\right],
$$

with

$$
\alpha(0)=\sqrt{\frac{1}{B}},\qquad
\alpha(k)=\sqrt{\frac{2}{B}}\;\text{for }k>0.
$$

The DC term $C[0,0]$ is excluded. Remaining coefficients are ordered by JPEG-style zig-zag traversal; the default uses 15 non-DC coefficients from each 4×4 block.

### 4. Default keyed spread-transform dither modulation

The selected scalar DCT coefficients are packed into a vector $\mathbf{x}$. A secret key $K$ is expanded with HMAC-SHA-256 into independent domains for:

- a carrier permutation $\pi_K$;
- balanced pseudo-noise sequences $\mathbf{p}_i\in\{-1,+1\}^{N_c}$;
- subtractive dithers $d_i\in[0,\Delta)$;
- payload whitening streams.

For coded bit $b_i\in\{0,1\}$, take the keyed chip

$$
\mathbf{x}_i =
\left[x_{\pi_K(iN_c)},\ldots,x_{\pi_K(iN_c+N_c-1)}\right]^T
$$

and its PN projection

$$
z_i = \frac{1}{N_c}\mathbf{p}_i^T\mathbf{x}_i.
$$

The two parity lattices are

$$
\Lambda_b(d_i)=\left\{d_i+\Delta(2k+b):k\in\mathbb{Z}\right\}.
$$

Embed by finding the nearest point in the lattice selected by the bit,

$$
z_i' = Q_{b_i}(z_i-d_i)+d_i,
$$

then distribute the projection error across the chip:

$$
\boxed{\mathbf{x}_i' = \mathbf{x}_i + (z_i'-z_i)\mathbf{p}_i.}
$$

Because $\|\mathbf{p}_i\|_2^2=N_c$, the new normalized projection is exactly $z_i'$. The extractor repeats the keyed projection and computes a soft score from the distance to the even and odd QIM lattices. Those scores feed the Viterbi decoder; no original cover is needed.

The main strength/capacity controls are the quantizer step $\Delta$, chip length $N_c$, DWT level, sub-band set, block size, and selected coefficients per block.

### 5. Optional DWT-DCT-SVD backend

For every selected DCT block,

$$
C = U\Sigma V^T,
\qquad
\Sigma = \operatorname{diag}(\sigma_0,\sigma_1,\ldots).
$$

A classical additive SVD watermark changes the singular values as

$$
\boxed{\Sigma_{\text{watermarked}} = \Sigma + \alpha W.}
$$

The implemented `svd-qim` backend replaces direct addition with keyed parity QIM on a chosen singular value:

$$
\sigma_k' = \underset{q\in\Lambda_b(d)}{\operatorname{argmin}}\;|\sigma_k-q|,
\qquad
C' = U\Sigma'V^T.
$$

This preserves blind bit decisions and constrains non-leading singular values to remain ordered. It is included for research comparisons; the STDM backend is the robust default.

### 6. Framing, ECC, and authentication

The logical payload is framed as

```text
magic | version | flags | payload_length | CRC32 | payload | HMAC_tag
```

Header and body are independently whitened with key-derived bit streams. The default channel code is a terminated rate-1/2 convolutional code with constraint length $K=7$ and generators

$$
(g_0,g_1)=(171_8,133_8).
$$

Extraction uses a soft-decision Viterbi decoder. CRC detects accidental corruption; the truncated HMAC rejects wrong keys and forged payloads. This is payload authentication—not media encryption, DRM, or a public ownership proof.

### 7. Inverse pipeline

Modified coefficients are inverse-DCT transformed, reassembled into `LH₃`/`HL₃`, and passed through the inverse multi-level DWT. After cropping the transform padding, the luminance plane is clipped, rounded, and merged with the untouched chroma/alpha channels.

The same generator creates `assets/residual_x10.png`, showing the signed embedding residual magnified by 10 and centered at gray.

## Spatio-temporal video watermarking

The video pipeline is a separable 3D transform: a one-level Haar transform along time followed by the spatial image transform inside each temporal low-pass plane.

For pairs of luminance frames,

$$
L_t = \frac{Y_{2t}+Y_{2t+1}}{\sqrt{2}},
\qquad
H_t = \frac{Y_{2t}-Y_{2t+1}}{\sqrt{2}}.
$$

The watermark is embedded into every selected $L_t$, while $H_t$ is retained. Reconstruction uses

$$
\widehat{Y}_{2t}=\frac{\widehat{L}_t+H_t}{\sqrt{2}},
\qquad
\widehat{Y}_{2t+1}=\frac{\widehat{L}_t-H_t}{\sqrt{2}}.
$$

This makes paired frames share the same low-frequency watermark perturbation, reducing temporal flicker. The default video preset raises spatial $\Delta$ from 12 to 16 before temporal reconstruction to compensate for attenuation and inter-frame quantization.

Frames are processed in configurable temporal windows (default 8). The encoder can embed every $G$-th group and forces output keyframes/GOP boundaries at the same group interval. During extraction:

1. each complete temporal window is Haar-decomposed;
2. soft STDM scores are recovered from every temporal low-pass plane;
3. scores are clipped and averaged across frames;
4. temporal starting phases are searched after frame deletion/rate conversion;
5. the candidate with the strongest valid header/CRC/HMAC evidence is returned.

This repetition and voting strategy is robust to ordinary H.264/H.265 transcoding, isolated frame loss, and moderate frame-rate conversion. It does not guarantee survival after arbitrary editing, severe cropping, scene replacement, or aggressive generational compression.

## Configuration guide

### Image configuration

| Field | Default | Effect |
|---|---:|---|
| `algorithm` | `stdm` | Robust PN/STDM default; `svd-qim` is the research alternative. |
| `wavelet` | `haar` | Spatial wavelet. PyWavelets enables additional discrete wavelets. |
| `dwt_level` | `3` | Embeds in level-$L$ details; higher levels are coarser and lower-capacity. |
| `block_size` | `4` | DCT block width/height. |
| `delta` | `12.0` | QIM spacing; larger is more robust but less imperceptible. |
| `subbands` | `LH,HL` | Orientation bands carrying the watermark. |
| `spread_chip_length` | `8` | DCT coefficients per physical bit; larger improves spreading but lowers capacity. |
| `spread_coefficients_per_block` | `15` | Non-DC zig-zag coefficients available from each 4×4 block. |
| `ecc_mode` | `convolutional` | Soft Viterbi channel code; `repetition` is simpler but weaker. |
| `auth_tag_bytes` | `8` | Truncated HMAC length. |

### Video configuration

| Field | Default | Effect |
|---|---:|---|
| `image.delta` | `16.0` | Stronger spatial step before temporal inverse transform. |
| `temporal_window` | `8` | Frames per separable 3D transform group. |
| `embed_group_interval` | `1` | Embed every $G$-th temporal group. |
| `codec` | `libx264` | FFmpeg output encoder; `libx265` is supported. |
| `crf` | `18` | Output quality/size control passed to the encoder. |
| `align_output_keyframes` | `true` | Force GOP/keyframe interval to match embedded temporal groups. |
| `extraction_max_samples` | `64` | Upper bound on low-pass score vectors used for voting. |
| `search_temporal_phase` | `true` | Search offsets $0,\ldots,T-1$ after temporal attacks. |

