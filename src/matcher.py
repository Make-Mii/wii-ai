import cv2
import torch

import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.jit._serialization"
)
import clip
import numpy as np
from PIL import Image

from utils import standardize_crop_for_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load precomputed embeddings
GALLERY_PATH = "mii_galleries.pt"
galleries = torch.load(GALLERY_PATH, map_location=device)

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


def match_feature(crop_input, category: str, top_k: int = 1):
    if category not in galleries:
        raise ValueError(
            f"Category '{category}' not found. Available: {list(galleries.keys())}"
        )

    # Use the standardized preprocessor
    gray_img = standardize_crop_for_clip(crop_input)
    tensor = preprocess(gray_img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embed = model.encode_image(tensor)
        query_embed /= query_embed.norm(dim=-1, keepdim=True)

        cat_embeddings = galleries[category]["embeddings"].to(device)
        filenames = galleries[category]["filenames"]

        scores = (query_embed @ cat_embeddings.T).squeeze(0)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(filenames)))

    return [
        (filenames[idx.item()], float(score.item()))
        for score, idx in zip(top_scores, top_indices)
    ]


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
