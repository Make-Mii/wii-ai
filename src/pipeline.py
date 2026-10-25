import numpy as np
from PIL import Image

from matcher import match_features_batch, match_color
from mii_payload import build_libmii_state, build_official_74byte_mii, send_mii_packet
from preview_mii import render_debug_mii

COLOR_CATEGORIES = {"hair", "eyebrows", "eyes", "mouths", "heads", "skin"}
CATEGORY_ALIASES = {
    "head": "heads",
    "face": "heads",
    "skin": "heads",
    "eye": "eyes",
    "eyes_single": "eyes",
    "eyebrow": "eyebrows",
    "nose": "noses",
    "mouth": "mouths",
    "lips": "mouths",
}


def process_segmented_face(
    crops: dict,
    wii_ip: str = None,
    port: int = 8080,
    format_type: str = "official",
    has_glasses=False,
) -> dict:

    if has_glasses:
        crops["glasses"] = Image.open("../mii_parts/transparent/glasses/glasses_06.png")

    # 1. Normalize category names and drop empty crops
    normalized = {}
    for category, crop in crops.items():
        if crop is None:
            continue
        cat = category.lower()
        cat = CATEGORY_ALIASES.get(cat, cat)
        normalized[cat] = crop

    # 2. One batched CLIP pass for every feature
    matches = match_features_batch(normalized, top_k=1)

    # 3. Colors + results
    matched_results = {}
    for cat, crop in normalized.items():
        best_asset, similarity = matches[cat][0]
        matched_entry = {"asset": best_asset, "similarity": similarity}

        if cat in COLOR_CATEGORIES:
            if isinstance(crop, Image.Image):
                rgb_arr = np.array(crop.convert("RGB"))
            else:
                rgb_arr = crop[:, :, :3] if crop.shape[-1] >= 3 else crop
            matched_entry["color_id"] = match_color(rgb_arr, category=cat)

        matched_results[cat] = matched_entry
        print(
            f"[{cat.upper():8s}] Matched: {best_asset:12s} "
            f"(Score: {similarity:.3f}) | Color ID: {matched_entry.get('color_id', 'N/A')}"
        )

    if wii_ip:
        send_mii_packet(
            matched_results, wii_ip=wii_ip, port=port, format_type=format_type
        )

    render_debug_mii(matched_results)
    return matched_results
