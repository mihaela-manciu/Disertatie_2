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


def get_inverse_freq_weights(stats, smoothing=0.1, max_weight=50.0):
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
        return_paste_stats=False,
        normalize=True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.enable_copy_paste = enable_copy_paste
        self.copy_paste_prob = copy_paste_prob
        self.prefer_plastic = prefer_plastic
        self.return_paste_stats = return_paste_stats
        self.normalize = normalize
        self.patches_dir = os.path.join(root_dir, "patches")

        split_file = os.path.join(root_dir, "splits", f"{split}_X.txt")
        with open(split_file, "r", encoding="utf-8") as f:
            self.image_names = [line.strip() for line in f.readlines()]

        self.class_mapping = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3}

        self._channel_mean = None
        self._channel_std = None
        if normalize:
            stats = load_channel_stats(root_dir)
            self._channel_mean = np.array(stats["mean"], dtype=np.float32)
            self._channel_std = np.array(stats["std"], dtype=np.float32)

        self._debris_flags = None
        self._plastic_flags = None

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
        scale = _detect_data_scale(self.root_dir)
        if scale != 1.0:
            img = img / scale
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

    def apply_copy_paste(self, img1, mask1):
        pasted_pixels = 0
        if random.random() > self.copy_paste_prob:
            return img1, mask1, pasted_pixels

        candidates = list(range(len(self.image_names)))
        random.shuffle(candidates)

        for idx2 in candidates[: min(30, len(candidates))]:
            img2, mask2 = self._load_image_and_mask(idx2)

            if self.prefer_plastic and np.any(mask2 == 1):
                donor_mask = mask2 == 1
            elif np.any((mask2 == 1) | (mask2 == 2)):
                donor_mask = (mask2 == 1) | (mask2 == 2)
            else:
                continue

            donor_ys, donor_xs = np.where(donor_mask)
            if len(donor_ys) == 0:
                continue

            y_min, y_max = donor_ys.min(), donor_ys.max()
            x_min, x_max = donor_xs.min(), donor_xs.max()
            patch_h = y_max - y_min + 1
            patch_w = x_max - x_min + 1

            water_mask = mask1 == 0
            h, w = mask1.shape
            if patch_h > h or patch_w > w:
                continue

            placed = False
            for _ in range(15):
                ry = random.randint(0, h - patch_h)
                rx = random.randint(0, w - patch_w)
                target_region = water_mask[ry:ry + patch_h, rx:rx + patch_w]
                local_ys = donor_ys - y_min
                local_xs = donor_xs - x_min
                if target_region[local_ys, local_xs].all():
                    dst_ys = ry + local_ys
                    dst_xs = rx + local_xs
                    img1[dst_ys, dst_xs, :] = img2[donor_ys, donor_xs, :]
                    mask1[dst_ys, dst_xs] = mask2[donor_ys, donor_xs]
                    pasted_pixels = len(donor_ys)
                    placed = True
                    break

            if placed:
                break

        return img1, mask1, pasted_pixels

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
