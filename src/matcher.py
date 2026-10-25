import cv2
import torch
import os

import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.jit._serialization"
)
import clip
import numpy as np
from PIL import Image

from utils import standardize_crop_for_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count())
model, preprocess = clip.load("ViT-B/32", device=device)

# Load precomputed embeddings
GALLERY_PATH = "mii_galleries.pt"
galleries = torch.load(GALLERY_PATH, map_location=device)
for cat in galleries.values():
    cat["embeddings"] = cat["embeddings"].to(device).float()

# Exact hex values from the Libmii specification
MII_PALETTES = {
    "skin": [0xECCFBD, 0xF7BC7D, 0xD78A48, 0xF5B189, 0x995122, 0x563010],
    "hair": [
        0x111111,
        0x332222,
        0x441111,
        0xBB6644,
        0x8888AA,
        0x443322,
        0x996644,
        0xDDBB99,
    ],
    "eyes": [0x000000, 0x778887, 0x7E6355, 0x888940, 0x6A84D0, 0x409B5A],
    "mouths": [0xC76C46, 0xE44E3A, 0xD88789],
    "glasses": [0x626D6C, 0x85703A, 0xAB4E37, 0x426996, 0xB97F27, 0xBDBFB9],
}

# Eyebrows, beards, and mustaches all inherit the hair color palette
MII_PALETTES["eyebrows"] = MII_PALETTES["hair"]
MII_PALETTES["beards"] = MII_PALETTES["hair"]
MII_PALETTES["mustaches"] = MII_PALETTES["hair"]


# Convert 24 bit hex integer to 1x3 float32 CIELAB vector
def _hex_to_lab(hex_val):
    rgb = np.array(
        [(hex_val >> 16) & 0xFF, (hex_val >> 8) & 0xFF, hex_val & 0xFF], dtype=np.uint8
    )
    return (
        cv2.cvtColor(rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)
        .reshape(3)
        .astype(np.float32)
    )


# Precompute LAB values once at module import
MII_PALETTES_LAB = {
    category: [_hex_to_lab(c) for c in hex_list]
    for category, hex_list in MII_PALETTES.items()
}


def match_features_batch(crops: dict, top_k: int = 1):
    cats = list(crops.keys())
    tensors = torch.stack(
        [preprocess(standardize_crop_for_clip(crops[c])) for c in cats]
    ).to(device)

    with torch.inference_mode():
        embeds = model.encode_image(tensors)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)

        results = {}
        for cat, emb in zip(cats, embeds):
            gallery = galleries[cat]
            scores = emb @ gallery["embeddings"].T
            vals, idxs = torch.topk(scores, k=min(top_k, len(gallery["filenames"])))
            results[cat] = [
                (gallery["filenames"][i], v)
                for v, i in zip(vals.tolist(), idxs.tolist())
            ]
    return results


def match_color(
    crop_rgb: np.ndarray, category: str, alpha_mask: np.ndarray = None
) -> int:
    """
    Finds the closest Mii palette index for a given feature crop.

    Args:
        crop_rgb: np.ndarray (H, W, 3) in RGB format.
        category: 'hair', 'eyes', 'skin', 'mouths', 'eyebrows', etc.
        alpha_mask: Optional boolean/binary mask (H, W) where 1/True represents feature pixels.

    Returns:
        int: The integer index (0 to N-1) corresponding to the libmii palette.
    """
    # Normalize category aliases
    if category in ["heads", "face", "skin"]:
        palette_key = "skin"
    elif category in ["lips", "mouth"]:
        palette_key = "mouths"
    elif category in MII_PALETTES_LAB:
        palette_key = category
    else:
        raise ValueError(
            f"No color palette defined for category '{category}'. Valid: {list(MII_PALETTES_LAB.keys())}"
        )

    # 1. Isolate the pixels to evaluate
    if alpha_mask is not None and np.any(alpha_mask > 0):
        pixels = crop_rgb[alpha_mask > 0]
    else:
        # Special sampling rule for eyes:
        # The center of an eye is the iris/pupil, while the outer corners are sclera (white)
        h, w = crop_rgb.shape[:2]
        if palette_key == "eyes":
            # Sample central 30% to hit the iris, ignoring the sclera (white of eye)
            pixels = crop_rgb[
                int(h * 0.35) : int(h * 0.65), int(w * 0.35) : int(w * 0.65)
            ].reshape(-1, 3)
        else:
            # General central 50% box to avoid background edge bleed
            pixels = crop_rgb[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].reshape(-1, 3)

    if len(pixels) == 0:
        return 0

    # 2. Extract representative pigment via median (rejects specular glares and harsh shadows)
    median_rgb = np.median(pixels, axis=0).astype(np.uint8)
    sample_lab = (
        cv2.cvtColor(median_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)
        .reshape(3)
        .astype(np.float32)
    )

    # 3. Euclidean distance (Delta E) in CIELAB
    palette_labs = MII_PALETTES_LAB[palette_key]
    distances = [np.linalg.norm(sample_lab - target_lab) for target_lab in palette_labs]

    return int(np.argmin(distances))
