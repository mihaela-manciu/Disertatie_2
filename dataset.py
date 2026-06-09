"""
MARIDA dataset with 14-channel input (11 Sentinel-2 bands + NDVI + FDI + PI).

Key changes vs Disertatie_Identificare-deseuri-marine:
  - 14 channels instead of 13 (added Plastic Index)
  - Improved rarity-aware sampler with configurable class-level boost
  - Per-patch debris flag caching for fast sampler construction
"""

import json
import os
import random

import albumentations as A
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


NUM_CHANNELS = 14   # 11 S2 bands + NDVI + FDI + PI

_STATS_FILENAME = "channel_stats_14ch.json"

_DATA_SCALE_CACHE = {}


def _detect_data_scale(root_dir, num_samples=10):
    """
    Auto-detect if MARIDA patches store DN values (0-10000) or reflectance
    (0-1).  Reads a few sample images and checks value ranges.
    """
    if root_dir in _DATA_SCALE_CACHE:
        return _DATA_SCALE_CACHE[root_dir]

    split_file = os.path.join(root_dir, "splits", "train_X.txt")
    if not os.path.isfile(split_file):
        _DATA_SCALE_CACHE[root_dir] = 10000.0
        return 10000.0

    with open(split_file, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines()]

    patches_dir = os.path.join(root_dir, "patches")
    max_vals = []

    for name in names:
        if len(max_vals) >= num_samples:
            break
        name = name.strip()
        folder = name.rsplit("_", 1)[0]
        path = os.path.join(patches_dir, folder, f"{name}.tif")
        if not os.path.exists(path) and not name.startswith("S2_"):
            path = os.path.join(patches_dir, f"S2_{folder}", f"S2_{name}.tif")
        if not os.path.exists(path):
            continue
        try:
            with rasterio.open(path) as src:
                raw = src.read().astype(np.float32)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            pos = raw[raw > 0]
            if len(pos) > 100:
                max_vals.append(float(np.percentile(pos, 99)))
        except Exception:
            continue

    if not max_vals:
        print("[data scale] Could not sample images; defaulting to DN scaling (/10000)")
        _DATA_SCALE_CACHE[root_dir] = 10000.0
        return 10000.0

    p99_median = float(np.median(max_vals))
    if p99_median > 10.0:
        scale = 10000.0
        print(f"[data scale] DN values detected (p99 median={p99_median:.1f}) "
              f"— dividing by 10000")
    else:
        scale = 1.0
        print(f"[data scale] Reflectance values detected (p99 median={p99_median:.6f}) "
              f"— no scaling needed")

    _DATA_SCALE_CACHE[root_dir] = scale
    return scale


def _compute_spectral_indices(raw_11ch):
    """
    Compute NDVI, FDI, and PI from 11-band Sentinel-2 reflectance (0–1 range).

    Returns (ndvi, fdi, pi), each of shape (H, W).

    Band indices (0-based in raw_11ch):
      0=B01, 1=B02, 2=B03, 3=B04(Red), 4=B05, 5=B06(RE2), 6=B07,
      7=B08(NIR), 8=B8A, 9=B11(SWIR1), 10=B12(SWIR2)
    """
    b4_red = raw_11ch[3]
    b6_re2 = raw_11ch[5]
    b8_nir = raw_11ch[7]
    b11_swir1 = raw_11ch[9]

    ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + 1e-8)
    fdi = b8_nir - (b6_re2 + (b11_swir1 - b6_re2) * 10 * 0.1873)
    pi = b8_nir / (b8_nir + b4_red + 1e-8)

    return ndvi, fdi, pi


def compute_channel_stats(root_dir, split="train", max_samples=None):
    """
    Per-channel mean/std for the 14-channel representation.
    Also computes class pixel counts for inverse-frequency weighting.
    Auto-detects data scale (DN vs reflectance).
    """
    scale = _detect_data_scale(root_dir)

    split_file = os.path.join(root_dir, "splits", f"{split}_X.txt")
    with open(split_file, "r", encoding="utf-8") as f:
        image_names = [line.strip() for line in f.readlines()]

    if max_samples is not None:
        image_names = image_names[:max_samples]

    patches_dir = os.path.join(root_dir, "patches")
    class_mapping = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3}

    channel_sum = np.zeros(NUM_CHANNELS, dtype=np.float64)
    channel_sq_sum = np.zeros(NUM_CHANNELS, dtype=np.float64)
    pixel_count = 0
    loaded_count = 0
    class_pixel_counts = np.zeros(4, dtype=np.int64)

    print(f"Computing 14-channel statistics from {len(image_names)} {split} images "
          f"(scale={scale})…")
    for i, img_base in enumerate(image_names):
        img_base = img_base.strip()
        folder_name = img_base.rsplit("_", 1)[0]
        img_path = os.path.join(patches_dir, folder_name, f"{img_base}.tif")
        mask_path = os.path.join(patches_dir, folder_name, f"{img_base}_cl.tif")

        if not os.path.exists(img_path):
            if not img_base.startswith("S2_"):
                img_base_s2 = "S2_" + img_base
                folder_name_s2 = "S2_" + folder_name
                alt_img = os.path.join(patches_dir, folder_name_s2, f"{img_base_s2}.tif")
                alt_mask = os.path.join(patches_dir, folder_name_s2, f"{img_base_s2}_cl.tif")
                if os.path.exists(alt_img):
                    img_path, mask_path = alt_img, alt_mask

        try:
            with rasterio.open(img_path) as src:
                raw = src.read().astype(np.float32)
        except Exception:
            continue

        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
        if scale != 1.0:
            raw = raw / scale
        raw = np.clip(raw, 0.0, 1.0)

        ndvi, fdi, pi = _compute_spectral_indices(raw)
        img14 = np.concatenate([raw, ndvi[None], fdi[None], pi[None]], axis=0)

        n_pixels = img14.shape[1] * img14.shape[2]
        channel_sum += img14.reshape(NUM_CHANNELS, -1).sum(axis=1)
        channel_sq_sum += (img14.reshape(NUM_CHANNELS, -1) ** 2).sum(axis=1)
        pixel_count += n_pixels
        loaded_count += 1

        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            mapped = np.zeros_like(mask, dtype=np.int64)
            for old_cls, new_cls in class_mapping.items():
                mapped[mask == old_cls] = new_cls
            for cls_id in range(4):
                class_pixel_counts[cls_id] += int((mapped == cls_id).sum())
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(image_names)} images processed…")

    if pixel_count == 0:
        raise RuntimeError(f"No images loaded from {root_dir}. Check patch paths.")

    mean = channel_sum / pixel_count
    std = np.sqrt(np.maximum(channel_sq_sum / pixel_count - mean ** 2, 0.0))
    std = np.maximum(std, 1e-6)

    total_pixels = int(class_pixel_counts.sum())
    class_freqs = class_pixel_counts / max(total_pixels, 1)

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_channels": NUM_CHANNELS,
        "data_scale": scale,
        "class_pixel_counts": class_pixel_counts.tolist(),
        "class_frequencies": class_freqs.tolist(),
        "total_pixels": total_pixels,
        "num_images": loaded_count,
    }

    out_path = os.path.join(root_dir, _STATS_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Channel statistics saved to {out_path} ({loaded_count}/{len(image_names)} images)")
    print(f"  Mean: {[f'{m:.6f}' for m in mean]}")
    print(f"  Std:  {[f'{s:.6f}' for s in std]}")
    print(f"  Class freqs: {[f'{fr:.6f}' for fr in class_freqs]}")
    return stats


def load_channel_stats(root_dir):
    path = os.path.join(root_dir, _STATS_FILENAME)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        if stats.get("num_channels", 13) != NUM_CHANNELS:
            print(f"[stats] Found {stats.get('num_channels')} channels, "
                  f"need {NUM_CHANNELS}. Recomputing…")
            return compute_channel_stats(root_dir)
        detected_scale = _detect_data_scale(root_dir)
        stored_scale = stats.get("data_scale", 10000.0)
        if abs(stored_scale - detected_scale) > 0.01:
            print(f"[stats] Data scale changed ({stored_scale} → {detected_scale}). "
                  f"Recomputing…")
            print(f"[stats] WARNING: old checkpoints were trained with wrong "
                  f"normalization. Use --no-resume to retrain from scratch.")
            return compute_channel_stats(root_dir)
        return stats
    return compute_channel_stats(root_dir)


def get_inverse_freq_weights(stats, smoothing=0.05, max_weight=50.0):
    freqs = np.array(stats["class_frequencies"], dtype=np.float64)
    freqs = np.maximum(freqs, 1e-6)
    inv = 1.0 / freqs
    inv = inv / inv.sum()
    weights = (1.0 - smoothing) * inv + smoothing * np.ones_like(inv)
    weights = weights / weights.min()
    if max_weight is not None:
        weights = np.minimum(weights, max_weight)
    return weights.tolist()


class MARIDADataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        *,
        enable_copy_paste=True,
        copy_paste_prob=0.5,
        prefer_plastic=True,
        copy_paste_min_pixels=0,
        copy_paste_max_instances=1,
        copy_paste_scale_range=(1.0, 1.0),
        copy_paste_max_scale=4.0,
        copy_paste_max_attempts=3,
        return_paste_stats=False,
        normalize=True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.enable_copy_paste = enable_copy_paste
        self.copy_paste_prob = copy_paste_prob
        self.prefer_plastic = prefer_plastic
        self.copy_paste_min_pixels = int(copy_paste_min_pixels)
        self.copy_paste_max_instances = max(1, int(copy_paste_max_instances))
        self.copy_paste_scale_range = tuple(copy_paste_scale_range)
        self.copy_paste_max_scale = float(copy_paste_max_scale)
        self.copy_paste_max_attempts = max(1, int(copy_paste_max_attempts))
        self.return_paste_stats = return_paste_stats
        self.normalize = normalize
        self.patches_dir = os.path.join(root_dir, "patches")

        split_file = os.path.join(root_dir, "splits", f"{split}_X.txt")
        with open(split_file, "r", encoding="utf-8") as f:
            self.image_names = [line.strip() for line in f.readlines()]

        self.class_mapping = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3}

        self._data_scale = _detect_data_scale(root_dir)

        self._channel_mean = None
        self._channel_std = None
        if normalize:
            stats = load_channel_stats(root_dir)
            self._channel_mean = np.array(stats["mean"], dtype=np.float32)
            self._channel_std = np.array(stats["std"], dtype=np.float32)

        self._debris_flags = None
        self._plastic_flags = None
        self._debris_donor_indices = None
        self._plastic_donor_indices = None

    def __len__(self):
        return len(self.image_names)

    def _cache_debris_flags(self):
        """Precompute which patches contain debris / plastic for fast sampling."""
        if self._debris_flags is not None:
            return
        print("Caching debris/plastic flags for sampler…")
        self._debris_flags = []
        self._plastic_flags = []
        for idx in range(len(self)):
            try:
                _, mask = self._load_image_and_mask(idx)
                self._debris_flags.append(bool(np.any((mask == 1) | (mask == 2))))
                self._plastic_flags.append(bool(np.any(mask == 1)))
            except Exception:
                self._debris_flags.append(False)
                self._plastic_flags.append(False)
        self._debris_donor_indices = [
            i for i, flag in enumerate(self._debris_flags) if flag
        ]
        self._plastic_donor_indices = [
            i for i, flag in enumerate(self._plastic_flags) if flag
        ]

    def _find_donor_with_debris(self):
        """Pick a random patch with foreground debris; prefer plastic donors when enabled."""
        self._cache_debris_flags()
        if self.prefer_plastic and self._plastic_donor_indices:
            return random.choice(self._plastic_donor_indices)
        if self._debris_donor_indices:
            return random.choice(self._debris_donor_indices)
        return random.randint(0, len(self) - 1)

    def has_debris(self, idx):
        if self._debris_flags is not None:
            return self._debris_flags[idx]
        _, mask = self._load_image_and_mask(idx)
        return bool(np.any((mask == 1) | (mask == 2)))

    def has_plastic(self, idx):
        if self._plastic_flags is not None:
            return self._plastic_flags[idx]
        _, mask = self._load_image_and_mask(idx)
        return bool(np.any(mask == 1))

    def _load_image_and_mask(self, idx):
        img_base = self.image_names[idx].strip()
        folder_name = img_base.rsplit("_", 1)[0]

        img_path = os.path.join(self.patches_dir, folder_name, f"{img_base}.tif")
        mask_path = os.path.join(self.patches_dir, folder_name, f"{img_base}_cl.tif")

        if not os.path.exists(img_path):
            if not img_base.startswith("S2_"):
                img_base_s2 = "S2_" + img_base
                folder_name_s2 = "S2_" + folder_name
                img_path_s2 = os.path.join(
                    self.patches_dir, folder_name_s2, f"{img_base_s2}.tif"
                )
                mask_path_s2 = os.path.join(
                    self.patches_dir, folder_name_s2, f"{img_base_s2}_cl.tif"
                )
                if os.path.exists(img_path_s2):
                    img_path = img_path_s2
                    mask_path = mask_path_s2

        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)

        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        if self._data_scale != 1.0:
            img = img / self._data_scale
        img = np.clip(img, 0.0, 1.0)

        ndvi, fdi, pi = _compute_spectral_indices(img)
        img_14ch = np.concatenate((img, ndvi[None], fdi[None], pi[None]), axis=0)

        if self._channel_mean is not None:
            img_14ch = (img_14ch - self._channel_mean[:, None, None]) / self._channel_std[:, None, None]

        img_14ch = np.transpose(img_14ch, (1, 2, 0))  # (H, W, 14)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        mapped_mask = np.zeros_like(mask, dtype=np.int64)
        for old_cls, new_cls in self.class_mapping.items():
            mapped_mask[mask == old_cls] = new_cls

        return img_14ch, mapped_mask

    @staticmethod
    def _resize_fg_patch(roi_img, roi_mask, roi_fg, out_h, out_w):
        """Upscale donor fg patch — linear for image, nearest for class mask."""
        in_h, in_w = roi_fg.shape
        if in_h == out_h and in_w == out_w:
            return roi_img, roi_mask, roi_fg
        try:
            import cv2
            roi_img_r = cv2.resize(roi_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            roi_mask_r = cv2.resize(
                roi_mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST,
            )
            roi_fg_r = cv2.resize(
                roi_fg.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST,
            ) > 0
            return roi_img_r, roi_mask_r, roi_fg_r
        except ImportError:
            y_idx = (np.arange(out_h) * in_h / out_h).astype(np.int64)
            x_idx = (np.arange(out_w) * in_w / out_w).astype(np.int64)
            roi_img_r = roi_img[y_idx][:, x_idx]
            roi_mask_r = roi_mask[y_idx][:, x_idx]
            roi_fg_r = roi_fg[y_idx][:, x_idx]
            return roi_img_r, roi_mask_r, roi_fg_r

    def _paste_one_instance(self, img1, mask1, img2, mask2):
        if self.prefer_plastic:
            paste_mask = (mask2 == 1)
        else:
            paste_mask = (mask2 == 1) | (mask2 == 2)
        ys, xs = np.where(paste_mask)
        if len(ys) == 0:
            return 0
        h, w = mask1.shape
        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())
        roi_img = img2[ymin : ymax + 1, xmin : xmax + 1, :].copy()
        roi_mask = mask2[ymin : ymax + 1, xmin : xmax + 1].copy()
        roi_fg = paste_mask[ymin : ymax + 1, xmin : xmax + 1]
        box_h, box_w = roi_fg.shape
        if box_h < 1 or box_w < 1:
            return 0

        scale_lo, scale_hi = self.copy_paste_scale_range
        scale = random.uniform(scale_lo, scale_hi) if scale_hi > scale_lo else scale_lo
        if self.copy_paste_min_pixels > 0 and roi_fg.sum() > 0:
            area_scale = (self.copy_paste_min_pixels / roi_fg.sum()) ** 0.5
            scale = max(scale, min(area_scale, self.copy_paste_max_scale))

        new_h = max(1, min(h, int(round(box_h * scale))))
        new_w = max(1, min(w, int(round(box_w * scale))))
        if new_h != box_h or new_w != box_w:
            roi_img, roi_mask, roi_fg = self._resize_fg_patch(
                roi_img, roi_mask, roi_fg, new_h, new_w,
            )

        box_h, box_w = roi_fg.shape
        if box_h > h or box_w > w:
            return 0

        for _ in range(self.copy_paste_max_attempts):
            y0 = random.randint(0, h - box_h)
            x0 = random.randint(0, w - box_w)
            target_mask = mask1[y0 : y0 + box_h, x0 : x0 + box_w]
            paste_where = roi_fg & (target_mask == 0)
            if not paste_where.any():
                continue
            target_img = img1[y0 : y0 + box_h, x0 : x0 + box_w, :]
            target_img[paste_where] = roi_img[paste_where]
            target_mask[paste_where] = roi_mask[paste_where]
            img1[y0 : y0 + box_h, x0 : x0 + box_w, :] = target_img
            mask1[y0 : y0 + box_h, x0 : x0 + box_w] = target_mask
            return int(paste_where.sum())
        return 0

    def apply_copy_paste(self, img1, mask1):
        if random.random() > self.copy_paste_prob:
            return img1, mask1, 0
        img1 = img1.copy()
        mask1 = mask1.copy()
        pasted = 0
        for _ in range(self.copy_paste_max_instances):
            donor_idx = self._find_donor_with_debris()
            img2, mask2 = self._load_image_and_mask(donor_idx)
            pasted += self._paste_one_instance(img1, mask1, img2, mask2)
            if (
                self.copy_paste_min_pixels > 0
                and pasted >= self.copy_paste_min_pixels
            ):
                break
        return img1, mask1, pasted

    def __getitem__(self, idx):
        img, mask = self._load_image_and_mask(idx)
        pasted_pixels = 0
        total_pixels = int(mask.size)

        if self.split == "train" and self.enable_copy_paste:
            img, mask, pasted_pixels = self.apply_copy_paste(img, mask)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, np.ndarray):
            mask = np.rint(mask).astype(np.int64)
            mask = np.clip(mask, 0, 3)
        elif isinstance(mask, torch.Tensor):
            mask = mask.round().long().clamp(0, 3)

        if isinstance(img, torch.Tensor):
            img_tensor = img.float()
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))

        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.long()
        else:
            mask_tensor = torch.from_numpy(mask.astype(np.int64))

        if self.return_paste_stats and self.split == "train":
            return img_tensor, mask_tensor, {
                "pasted_pixels": pasted_pixels,
                "total_pixels": total_pixels,
            }
        return img_tensor, mask_tensor


def marida_pad_collate(batch):
    """Pad mixed 128/256 training samples to a common size for batching."""
    has_meta = len(batch[0]) == 3
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    metas = [item[2] for item in batch] if has_meta else None

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    channels = images[0].shape[0]
    batch_imgs = []
    batch_masks = []
    for img, mask in zip(images, masks):
        h, w = img.shape[1], img.shape[2]
        if h == max_h and w == max_w:
            batch_imgs.append(img)
            batch_masks.append(mask)
            continue
        padded = torch.zeros((channels, max_h, max_w), dtype=img.dtype)
        padded[:, :h, :w] = img
        batch_imgs.append(padded)
        padded_m = torch.zeros((max_h, max_w), dtype=mask.dtype)
        padded_m[:h, :w] = mask
        batch_masks.append(padded_m)

    out_imgs = torch.stack(batch_imgs, dim=0)
    out_masks = torch.stack(batch_masks, dim=0)
    if not has_meta:
        return out_imgs, out_masks

    if isinstance(metas[0], dict) and "pasted_pixels" in metas[0]:
        pasted = torch.tensor([int(m.get("pasted_pixels", 0)) for m in metas], dtype=torch.long)
        total = torch.tensor([int(m.get("total_pixels", 0)) for m in metas], dtype=torch.long)
        return out_imgs, out_masks, {"pasted_pixels": pasted, "total_pixels": total}
    return out_imgs, out_masks, metas


class MARIDACropMiningDataset(Dataset):
    """
    Wraps MARIDADataset with a mixed 128/256 schedule:
    - full_patch_prob: native patch (e.g. 256×256) for test distribution match
    - otherwise: fixed crop_size windows (debris-centered or random)
    """

    def __init__(
        self,
        base: MARIDADataset,
        crop_size: int = 128,
        crop_prob: float = 0.5,
        plastic_focus_prob: float = 0.75,
        full_patch_prob: float = 0.5,
    ):
        self.base = base
        self.crop_size = int(crop_size)
        self.crop_prob = float(crop_prob)
        self.plastic_focus_prob = float(plastic_focus_prob)
        self.full_patch_prob = float(full_patch_prob)

    def __len__(self):
        return len(self.base)

    def _cache_debris_flags(self):
        """Delegate to base — used by build_debris_weighted_sampler in main process."""
        return self.base._cache_debris_flags()

    @property
    def _debris_flags(self):
        return self.base._debris_flags

    @property
    def _plastic_flags(self):
        return self.base._plastic_flags

    def __getstate__(self):
        return {
            "base": self.base,
            "crop_size": self.crop_size,
            "crop_prob": self.crop_prob,
            "plastic_focus_prob": self.plastic_focus_prob,
            "full_patch_prob": self.full_patch_prob,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _to_hwc(self, img):
        if isinstance(img, torch.Tensor):
            return img.permute(1, 2, 0).numpy()
        return img

    def _fixed_window(self, img_hwc, mask, y0, x0):
        """Extract crop_size×crop_size window; pad if the source patch is smaller."""
        cs = self.crop_size
        h, w = mask.shape
        if h >= cs and w >= cs:
            y0 = int(max(0, min(y0, h - cs)))
            x0 = int(max(0, min(x0, w - cs)))
            return (
                img_hwc[y0:y0 + cs, x0:x0 + cs].copy(),
                mask[y0:y0 + cs, x0:x0 + cs].copy(),
            )
        channels = img_hwc.shape[2] if img_hwc.ndim == 3 else 1
        out_img = np.zeros((cs, cs, channels), dtype=img_hwc.dtype)
        out_mask = np.zeros((cs, cs), dtype=mask.dtype)
        hh, ww = min(h, cs), min(w, cs)
        out_img[:hh, :ww] = img_hwc[:hh, :ww]
        out_mask[:hh, :ww] = mask[:hh, :ww]
        return out_img, out_mask

    def _debris_centered_window(self, img_hwc, mask):
        cs = self.crop_size
        h, w = mask.shape
        if np.any(mask == 1) and random.random() < self.plastic_focus_prob:
            ys, xs = np.where(mask == 1)
        elif np.any((mask == 1) | (mask == 2)):
            ys, xs = np.where((mask == 1) | (mask == 2))
        else:
            y0 = random.randint(0, max(0, h - cs)) if h >= cs else 0
            x0 = random.randint(0, max(0, w - cs)) if w >= cs else 0
            return self._fixed_window(img_hwc, mask, y0, x0)
        idx = random.randint(0, len(ys) - 1)
        cy, cx = int(ys[idx]), int(xs[idx])
        return self._fixed_window(img_hwc, mask, cy - cs // 2, cx - cs // 2)

    def _random_window(self, img_hwc, mask):
        cs = self.crop_size
        h, w = mask.shape
        y0 = random.randint(0, max(0, h - cs)) if h >= cs else 0
        x0 = random.randint(0, max(0, w - cs)) if w >= cs else 0
        return self._fixed_window(img_hwc, mask, y0, x0)

    def __getitem__(self, idx):
        item = self.base[idx]
        if len(item) == 3:
            img, mask, meta = item
        else:
            img, mask = item
            meta = None
        if self.base.split == "train" and random.random() >= self.full_patch_prob:
            img_np = self._to_hwc(img)
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
            if random.random() < self.crop_prob:
                img_np, mask_np = self._debris_centered_window(img_np, mask_np)
            else:
                img_np, mask_np = self._random_window(img_np, mask_np)
            img = torch.from_numpy(img_np.transpose(2, 0, 1).astype(np.float32))
            mask = torch.from_numpy(mask_np.astype(np.int64)).clamp(0, 3)
        elif self.base.split == "train":
            if isinstance(mask, torch.Tensor):
                mask = mask.clamp(0, 3)
        if meta is not None:
            return img, mask, meta
        return img, mask


def build_debris_weighted_sampler(dataset, debris_boost=5.0, plastic_boost=10.0):
    """
    Oversample patches with debris; extra-boost patches with plastic.
    Uses cached flags for speed.
    """
    dataset._cache_debris_flags()
    weights = []
    print("Building weighted sampler (debris + plastic boost)…")
    for idx in range(len(dataset)):
        if dataset._plastic_flags[idx]:
            weights.append(plastic_boost)
        elif dataset._debris_flags[idx]:
            weights.append(debris_boost)
        else:
            weights.append(1.0)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_training_augmentation(enable=True):
    """Geometric-only augmentations — safe for multispectral satellite data."""
    if not enable:
        return A.Compose([ToTensorV2()])
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.3,
                mask_interpolation=0,
            ),
            ToTensorV2(),
        ]
    )


def get_validation_augmentation():
    return A.Compose([ToTensorV2()])


if __name__ == "__main__":
    cale_marida = r"D:\TAID\Disertatie\MARIDA"
    train_dataset = MARIDADataset(
        root_dir=cale_marida,
        split="train",
        transform=get_training_augmentation(),
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    print(f"Nr de batch-uri in antrenare: {len(train_loader)}")
    for images, masks in train_loader:
        print(f"Imagini: {images.shape}, măști: {masks.shape}")
        assert images.shape[1] == NUM_CHANNELS, f"Expected {NUM_CHANNELS} channels, got {images.shape[1]}"
        break
