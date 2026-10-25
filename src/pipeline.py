import numpy as np
from PIL import Image

from matcher import match_feature, match_color
from mii_payload import build_libmii_state, build_official_74byte_mii, send_mii_packet
from preview_mii import render_debug_mii

COLOR_CATEGORIES = {"hair", "eyebrows", "eyes", "mouths", "heads", "skin"}


def process_segmented_face(
    crops: dict, wii_ip: str = None, port: int = 8080, format_type: str = "official"
) -> dict:
    """
    Takes a dictionary of feature crops from Teammate 1's segmentation model,
    matches each feature to Mii assets, extracts colors, builds the payload,
    and optionally sends it directly to the Wii.

    Args:
        crops: dict mapping category names to PIL Images or NumPy RGB arrays.
        wii_ip: Optional string containing the target Wii's IP (e.g. "192.168.1.39").
        port: UDP port target on the Wii (default 8080).
        format_type: "official" (74-byte .mii blob for your teammate's code)
                     or "libmii" (38-byte flat struct).

    Returns:
        dict: The matched feature results dictionary.
    """
    matched_results = {}

    for category, crop in crops.items():
        if crop is None:
            continue

        # Standardize category name aliases
        cat = category.lower()
        if cat in ("head", "face", "skin"):
            cat = "heads"
        elif cat in ("eye", "eyes_single"):
            cat = "eyes"
        elif cat in ("eyebrow",):
            cat = "eyebrows"
        elif cat in ("nose",):
            cat = "noses"
        elif cat in ("mouth", "lips"):
            cat = "mouths"

        # 1. Match visual shape geometry via CLIP
        top_matches = match_feature(crop, category=cat, top_k=1)
        best_asset, similarity = top_matches[0]

        matched_entry = {"asset": best_asset, "similarity": similarity}

        # 2. Extract representative palette color in CIELAB
        if cat in COLOR_CATEGORIES:
            if isinstance(crop, Image.Image):
                rgb_arr = np.array(crop.convert("RGB"))
            else:
                rgb_arr = crop[:, :, :3] if crop.shape[-1] >= 3 else crop

            matched_entry["color_id"] = match_color(rgb_arr, category=cat)

        matched_results[cat] = matched_entry
        print(
            f"[{cat.upper():8s}] Matched: {best_asset:12s} (Score: {similarity:.3f}) | Color ID: {matched_entry.get('color_id', 'N/A')}"
        )

    # 3. If an IP was provided, transmit over the network using the chosen format
    if wii_ip:
        send_mii_packet(
            matched_results, wii_ip=wii_ip, port=port, format_type=format_type
        )

    render_debug_mii(matched_results)

    return matched_results
