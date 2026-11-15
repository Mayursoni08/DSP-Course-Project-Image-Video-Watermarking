# Algorithm reference

This document records implementation-level details needed to reproduce or extend the watermarking core. The README provides a pipeline overview and operating instructions.

## Coordinate and transform conventions

- OpenCV arrays are grayscale, BGR, or BGRA.
- Color inputs are transformed to YCrCb and only `Y` is modified.
- `core.transforms.dwt2` returns `(LL, (LH, HL, HH))` while internally mapping PyWavelets' detail arrays `(cH, cV, cD)` to those names.
- Detail levels returned by `multilevel_dwt2` are ordered from level 1 through level $L$; embedding modifies the last tuple.
- DCT and inverse DCT use orthonormal type-II transforms over the last two array axes.
- The default spatial multiple is $2^L B=32$.

## Default carrier count

For a 512×512 plane with $L=3$:

1. Each level-3 detail band has shape 64×64.
2. A 4×4 partition gives 256 blocks per band.
3. Two bands and 15 non-DC coefficients give $256\cdot2\cdot15=7680$ scalar carriers.
4. Chip length 8 gives $7680/8=960$ physical coded-bit carriers.
5. The 10-byte header is 80 logical bits. The terminated rate-1/2 code needs $2(80+6)=172$ carriers.
6. Remaining carriers: $960-172=788$.
7. The body can contain $\lfloor788/2\rfloor-6=388$ logical bits, or 48 complete bytes.
8. Removing the 8-byte HMAC tag leaves 40 payload bytes.

## Domain-separated key expansion

The user secret is first mapped to a 256-bit internal key:

$$
K'=\operatorname{SHA256}(\texttt{"dsp-invisible-watermark/v1"}\parallel 0\parallel K).
$$

Counter-mode HMAC-SHA-256 expands $K'$ under separate context strings for carrier permutation, dither, PN chips, header whitening, body whitening, and authentication. Domain separation prevents one generated stream from being reused for another role.

This construction is deterministic and suitable for keyed carrier generation. It is not presented as a replacement for a standardized encryption mode or password-based key derivation function. Production deployments should supply a uniformly random secret rather than a low-entropy password.

## Balanced PN chips

Every PN row contains exactly $N_c/2$ values of +1 and $N_c/2$ values of −1. Therefore its sample mean is zero, reducing DC bias in the selected coefficient population. The coefficient update is vectorized across all coded bits.

## Soft QIM score

For normalized de-dithered projection $x=(z-d)/\Delta$, let $e$ and $o$ be its nearest even and odd integers. The returned score is

$$
s=\operatorname{clip}(|x-e|-|x-o|,-1,1).
$$

Thus $s<0$ favors bit 0, $s>0$ favors bit 1, and $|s|$ is a bounded confidence proxy. Soft scores are more useful to the Viterbi decoder than hard lattice decisions.

## Convolutional coding

The encoder appends six zero tail bits, ensuring the trellis terminates in state zero. Each input bit emits two parity bits using generator polynomials $(171_8,133_8)$. The decoder minimizes a correlation-derived path metric and traces back from state zero.

Header and body are coded independently. If the header cannot be authenticated but the expected payload length is supplied through a manifest or CLI flag, the body can still be decoded for BER analysis; the result remains invalid unless header, CRC, and HMAC all pass.

## Geometric registration

Given attacked image $A$ and original reference $R$:

1. detect ORB keypoints/descriptors;
2. apply Lowe's ratio test to Hamming matches;
3. estimate an attacked-to-reference similarity transform with RANSAC;
4. if that fails, estimate a homography;
5. otherwise resize and run dense ECC affine alignment;
6. reflect-fill undefined borders.

Similarity is preferred over a full homography because the benchmark attacks use rotation, uniform scale, and translation, and lower-degree warps usually introduce less interpolation distortion.

## Video extraction phase search

Frame deletion or frame-rate conversion can shift the start of an 8-frame temporal window. The extractor evaluates phases $0\ldots7$. For each phase, all complete selected windows are decomposed. Soft scores from low-pass planes are clipped to $[-1,1]$, averaged, and decoded. Candidates are ranked lexicographically by authentication validity, header, CRC, HMAC, and confidence.

## Numerical behavior

All transform arithmetic is float64. Final image/frame output is rounded once and clipped to uint8. PyWavelets is preferred in installed environments. An exact orthonormal Haar fallback keeps the default mode operational when PyWavelets cannot be imported; non-Haar wavelets still require PyWavelets.
