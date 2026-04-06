"""
fundus_preprocessor.py
======================
Fundus image preprocessing pipeline for DR grading.

Removes dataset/client-specific bias so the model sees anatomy,
not camera artifacts. Apply identically during:
  - Training   : each client preprocesses its own images before local training
  - Inference  : uploaded images are preprocessed before model.predict()

Pipeline stages:
  1. Circular crop    — removes black border / vignette
  2. Ben Graham norm  — removes illumination gradient (most impactful)
  3. Green channel    — highest contrast for retinal vasculature
  4. CLAHE            — local contrast enhancement for fine lesions
  5. Gaussian denoise — reduces sensor noise
  6. Resize + pad     — fixed square output, preserves aspect ratio
  7. Per-image standardise — zero-mean, unit-variance

Also computes IID score per client — Jensen-Shannon divergence of the
label/intensity distribution vs the global population distribution,
used to track non-IID-ness in the federated research dashboard.

References:
  Ben Graham (2015 Kaggle DR winner)
  Gargeya & Leng 2017 (green channel for vessel contrast)
  Hagos & Kant 2019 (CLAHE for DR lesion enhancement)
  Zhou et al. 2023 MICCAI (domain-invariant preprocessing for FL-DR)
"""

import os
import math
import numpy as np

# cv2 is optional — graceful fallback when not installed
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


DEFAULT_CONFIG = {
    "output_size":       512,
    "clahe_clip":        2.0,
    "clahe_grid":        (8, 8),
    "ben_graham_sigma":  10,
    "ben_graham_scale":  4,
    "denoise_sigma":     0.3,
    "crop_threshold":    7,
    "use_green_channel": True,
    "use_ben_graham":    True,
    "use_clahe":         True,
    "use_denoise":       True,
    "standardise":       True,
    "output_channels":   3,
}


class FundusPreprocessor:
    """
    Stateless preprocessing pipeline. Thread-safe.

    Usage:
        prep   = FundusPreprocessor()
        img    = prep.process("/path/to/fundus.jpg")
        # img shape: (512, 512, 3), dtype float32, values ~N(0,1)

    Plug into your model:
        # PyTorch
        tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
        # Keras / TF
        tensor = np.expand_dims(img, 0)
    """

    def __init__(self, config: dict = None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        if not _CV2_AVAILABLE:
            raise ImportError(
                "opencv-python is required for image preprocessing.\n"
                "Install it with:  pip install opencv-python"
            )

    # ── Main entry point ────────────────────────────────────────────────────

    def process(self, source) -> np.ndarray:
        """
        Load and preprocess a fundus image.
        source: file path (str) or pre-loaded RGB numpy array (uint8).
        Returns: float32 ndarray (H, W, C).
        """
        img = self._load(source)
        img = self._circular_crop(img)
        img = self._resize_pad(img, self.cfg["output_size"])
        img = self._preprocess_channels(img)
        img = self._do_standardise(img)
        return img

    def process_batch(self, sources: list) -> np.ndarray:
        return np.stack([self.process(s) for s in sources])

    # ── Pipeline stages ──────────────────────────────────────────────────────

    def _load(self, source) -> np.ndarray:
        if isinstance(source, (str, os.PathLike)):
            img = cv2.imread(str(source))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.asarray(source, dtype=np.uint8)

    def _circular_crop(self, img: np.ndarray) -> np.ndarray:
        """Remove black vignette border — different cameras produce different sizes."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self.cfg["crop_threshold"], 255, cv2.THRESH_BINARY)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        coords  = cv2.findNonZero(mask)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        m = 5
        return img[max(0,y-m):min(img.shape[0],y+h+m),
                   max(0,x-m):min(img.shape[1],x+w+m)]

    def _resize_pad(self, img: np.ndarray, size: int) -> np.ndarray:
        """Resize preserving aspect ratio, pad to square with black."""
        h, w   = img.shape[:2]
        scale  = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_h, pad_w = size - new_h, size - new_w
        return cv2.copyMakeBorder(
            resized,
            pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    def _preprocess_channels(self, img: np.ndarray) -> np.ndarray:
        green = img[:, :, 1].copy()

        # Ben Graham: subtract blurred version to remove illumination gradient.
        # This is the single most effective step for cross-dataset normalisation.
        if self.cfg["use_ben_graham"]:
            sigma   = self.cfg["ben_graham_sigma"]
            scale   = self.cfg["ben_graham_scale"]
            blurred = cv2.GaussianBlur(green, (0, 0), sigma)
            green   = cv2.addWeighted(green, scale, blurred, -scale, 128)
            green   = np.clip(green, 0, 255).astype(np.uint8)

        # CLAHE: local contrast enhancement for microaneurysms / exudates
        if self.cfg["use_clahe"]:
            clahe = cv2.createCLAHE(
                clipLimit=self.cfg["clahe_clip"],
                tileGridSize=self.cfg["clahe_grid"]
            )
            green = clahe.apply(green)

        # Gentle denoise — preserves vessel edges
        if self.cfg["use_denoise"]:
            sigma = self.cfg["denoise_sigma"]
            ksize = max(3, int(sigma * 6) | 1)
            green = cv2.GaussianBlur(green, (ksize, ksize), sigma)

        if self.cfg["output_channels"] == 1:
            return green[:, :, np.newaxis].astype(np.float32)

        # Keep original R and B alongside processed green so ImageNet
        # pretrained backbones receive 3-channel input with preserved colour
        # info for hard exudate / NVD detection
        result = np.stack([img[:, :, 0], green, img[:, :, 2]], axis=-1)
        return result.astype(np.float32)

    def _do_standardise(self, img: np.ndarray) -> np.ndarray:
        """Per-image zero-mean unit-variance — works at inference without dataset stats."""
        if not self.cfg["standardise"]:
            return img / 255.0
        mean = img.mean()
        std  = img.std()
        return (img - mean) / max(std, 1e-6)

    # ── Quality check ────────────────────────────────────────────────────────

    def quality_check(self, source) -> dict:
        """Fast quality assessment — runs on upload before inference."""
        img  = self._load(source)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        brightness = float(gray.mean())
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        _, mask    = cv2.threshold(gray, self.cfg["crop_threshold"], 255, cv2.THRESH_BINARY)
        coverage   = float(mask.sum()) / (255 * gray.size)

        return {
            "brightness":   round(brightness, 1),
            "blur_score":   round(blur_score, 1),
            "coverage":     round(coverage, 3),
            "is_dark":      brightness < 40,
            "is_blurry":    blur_score < 50,
            "low_coverage": coverage < 0.3,
            "ok":           not (brightness < 40 or blur_score < 50 or coverage < 0.3),
        }

    # ── Visualise preprocessing stages (for the UI debug view) ───────────────

    def visualise_stages(self, source) -> dict:
        """Returns dict of intermediate uint8 images for display."""
        stages = {}
        img = self._load(source);          stages["original"] = img.copy()
        img = self._circular_crop(img);    stages["cropped"]  = img.copy()
        img = self._resize_pad(img, self.cfg["output_size"])

        green = img[:, :, 1].copy()
        stages["green_channel"] = cv2.merge([green, green, green])

        if self.cfg["use_ben_graham"]:
            sigma = self.cfg["ben_graham_sigma"]
            scale = self.cfg["ben_graham_scale"]
            blurred = cv2.GaussianBlur(green, (0, 0), sigma)
            bg = np.clip(cv2.addWeighted(green, scale, blurred, -scale, 128), 0, 255).astype(np.uint8)
            stages["ben_graham"] = cv2.merge([bg, bg, bg])
            green = bg

        if self.cfg["use_clahe"]:
            cl = cv2.createCLAHE(clipLimit=self.cfg["clahe_clip"],
                                 tileGridSize=self.cfg["clahe_grid"]).apply(green)
            stages["clahe"] = cv2.merge([cl, cl, cl])

        final = self.process(source)
        f_u8  = ((final - final.min()) / (final.max() - final.min() + 1e-6) * 255).astype(np.uint8)
        stages["final"] = f_u8
        return stages


# ── IID score computation ────────────────────────────────────────────────────

def compute_iid_score(client_label_counts: list, global_label_counts: list = None) -> float:
    """
    Compute Jensen-Shannon divergence between a client's label distribution
    and the global (pooled) distribution.

    Returns a score in [0, 1]:
      0.0 = perfectly IID (client mirrors global distribution)
      1.0 = maximally non-IID (client has entirely different classes)

    Args:
        client_label_counts: list of length n_classes, count per class for this client
        global_label_counts: list of length n_classes, global counts (optional;
                             if None uses uniform distribution as reference)
    """
    n = len(client_label_counts)
    p = np.array(client_label_counts, dtype=float)
    p = p / (p.sum() + 1e-9)

    if global_label_counts is not None:
        q = np.array(global_label_counts, dtype=float)
        q = q / (q.sum() + 1e-9)
    else:
        q = np.ones(n) / n   # uniform reference

    # Jensen-Shannon divergence (symmetric, bounded [0, log2])
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    # Normalise to [0, 1]
    return round(min(jsd / 1.0, 1.0), 4)


def simulate_client_label_distribution(client_idx: int, n_classes: int = 5,
                                        heterogeneity: float = 0.7) -> list:
    """
    Simulate a non-IID label distribution for a client using a Dirichlet draw.
    heterogeneity in [0, 1]: 0 = IID, 1 = maximally skewed.

    Used in TrainingWorker to generate realistic per-client distributions.
    """
    alpha = (1.0 - heterogeneity) * 5.0 + 0.1   # concentration param
    rng   = np.random.RandomState(client_idx * 7 + 13)
    probs = rng.dirichlet([alpha] * n_classes)
    total = 1000 + client_idx * 200
    return [int(p * total) for p in probs]
