"""Semi-blind geometric registration for crop/scale/rotation attacks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    image: NDArray[np.uint8]
    method: str
    inliers: int
    success: bool


def _gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def align_to_reference(
    attacked: NDArray[np.uint8],
    reference: NDArray[np.uint8],
    *,
    max_features: int = 6000,
    ratio_test: float = 0.78,
) -> RegistrationResult:
    """Warp an attacked image into the reference coordinate system.

    A RANSAC similarity transform is preferred because rotation/crop/scale
    attacks are usually well modeled by uniform scale, rotation, and
    translation; it introduces less resampling distortion than a full
    homography. Homography and dense ECC affine registration are fallbacks.
    """

    attacked_u8 = np.asarray(attacked, dtype=np.uint8)
    reference_u8 = np.asarray(reference, dtype=np.uint8)
    target_size = (reference_u8.shape[1], reference_u8.shape[0])
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=7)
    key_a, desc_a = orb.detectAndCompute(_gray(attacked_u8), None)
    key_r, desc_r = orb.detectAndCompute(_gray(reference_u8), None)

    if desc_a is not None and desc_r is not None and len(key_a) >= 8 and len(key_r) >= 8:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(desc_a, desc_r, k=2)
        good = [m for m, n in pairs if m.distance < ratio_test * n.distance]
        if len(good) >= 8:
            source_2d = np.float32([key_a[m.queryIdx].pt for m in good])
            destination_2d = np.float32([key_r[m.trainIdx].pt for m in good])
            affine, affine_mask = cv2.estimateAffinePartial2D(
                source_2d,
                destination_2d,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=5000,
                confidence=0.995,
                refineIters=25,
            )
            if affine is not None and np.isfinite(affine).all():
                warped = cv2.warpAffine(
                    attacked_u8,
                    affine,
                    target_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                inliers = int(affine_mask.sum()) if affine_mask is not None else 0
                return RegistrationResult(warped, "orb-similarity", inliers, True)

            source = source_2d.reshape(-1, 1, 2)
            destination = destination_2d.reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(
                source, destination, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.995
            )
            if homography is not None and np.isfinite(homography).all():
                warped = cv2.warpPerspective(
                    attacked_u8,
                    homography,
                    target_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                inliers = int(mask.sum()) if mask is not None else 0
                return RegistrationResult(warped, "orb-homography", inliers, True)

    resized = cv2.resize(attacked_u8, target_size, interpolation=cv2.INTER_LINEAR)
    template = _gray(reference_u8).astype(np.float32) / 255.0
    moving = _gray(resized).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
    try:
        cv2.findTransformECC(template, moving, warp, cv2.MOTION_AFFINE, criteria, None, 5)
        aligned = cv2.warpAffine(
            resized,
            warp,
            target_size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return RegistrationResult(aligned, "ecc-affine", 0, True)
    except cv2.error:
        return RegistrationResult(resized, "resize-only", 0, False)
