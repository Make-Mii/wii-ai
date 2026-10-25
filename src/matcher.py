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
        [
            preprocess(standardize_crop_for_clip(crops[c], enhance_texture=True))
            for c in cats
        ]
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
    if category in ["heads", "face", "skin"]:
        palette_key = "skin"
    elif category in ["lips", "mouth"]:
        palette_key = "mouths"
    elif category in MII_PALETTES_LAB:
        palette_key = category
    else:
        raise ValueError(f"No color palette defined for category '{category}'.")

    # 1. Strictly isolate foreground pixels (ignore [255, 255, 255] mask background)
    if alpha_mask is not None and np.any(alpha_mask > 0):
        pixels = crop_rgb[alpha_mask > 0]
    else:
        # A pixel is background if all channels are >= 250
        fg_mask = ~np.all(crop_rgb >= 250, axis=-1)
        pixels = crop_rgb[fg_mask]

    if len(pixels) < 10:
        return 0

    # 2. Convert valid pixels to CIELAB
    pixels_lab = (
        cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
        .reshape(-1, 3)
        .astype(np.float32)
    )

    # 3. Category-specific pixel filtering & perceptual channel weights
    if palette_key == "skin":
        # Filter out facial hair/beard shadows (bottom 30% L) and specular forehead glare (top 15% L)
        l_low, l_high = np.percentile(pixels_lab[:, 0], [30, 85])
        valid = (pixels_lab[:, 0] >= l_low) & (pixels_lab[:, 0] <= l_high)
        if np.any(valid):
            pixels_lab = pixels_lab[valid]
        # Skin tone depends primarily on Lightness (L), with softer chroma (A, B) weight
        # because Mii middle skin tones are artificially orange
        weights = np.array([1.2, 0.65, 0.65], dtype=np.float32)

    elif palette_key == "hair":
        # Filter out extreme specular shine from overhead lights and deep background shadow
        l_low, l_high = np.percentile(pixels_lab[:, 0], [20, 80])
        valid = (pixels_lab[:, 0] >= l_low) & (pixels_lab[:, 0] <= l_high)
        if np.any(valid):
            pixels_lab = pixels_lab[valid]
        # Standard perceptual balance for hair
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    elif palette_key == "eyes":
        # Filter out white sclera (L > 170) and black pupil/eyelashes (L < 35)
        valid = (pixels_lab[:, 0] > 35) & (pixels_lab[:, 0] < 170)
        if np.sum(valid) > 10:
            pixels_lab = pixels_lab[valid]
        # For iris color, hue/chroma (A, B) matters more than lighting
        weights = np.array([0.6, 1.3, 1.3], dtype=np.float32)

    else:
        # Mouths / Eyebrows / Glasses: standard Euclidean CIELAB Delta-E
        l_low, l_high = np.percentile(pixels_lab[:, 0], [20, 85])
        valid = (pixels_lab[:, 0] >= l_low) & (pixels_lab[:, 0] <= l_high)
        if np.any(valid):
            pixels_lab = pixels_lab[valid]
        weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    sample_lab = np.median(pixels_lab, axis=0)

    palette_labs = MII_PALETTES_LAB[palette_key]
    distances = [
        np.linalg.norm((sample_lab - target_lab) * weights)
        for target_lab in palette_labs
    ]

    return int(np.argmin(distances))
