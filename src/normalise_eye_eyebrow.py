import os
from PIL import Image

SRC = "renderMii_sprites"
OUT_CLIP = "mii_parts/clip_preview"
OUT_WII = "mii_parts/transparent"

os.makedirs(OUT_CLIP, exist_ok=True)
os.makedirs(OUT_WII, exist_ok=True)

eyebrows_lut = [
    1,
    3,
    14,
    15,
    11,
    10,
    0,
    6,
    8,
    4,
    13,
    12,
    2,
    19,
    16,
    18,
    22,
    9,
    21,
    5,
    17,
    7,
    20,
    23,
]
eyes_lut = [
    2,
    6,
    0,
    42,
    1,
    24,
    29,
    36,
    3,
    16,
    45,
    13,
    17,
    26,
    46,
    9,
    8,
    5,
    33,
    14,
    11,
    20,
    44,
    18,
    30,
    21,
    7,
    10,
    34,
    41,
    31,
    32,
    15,
    12,
    19,
    23,
    27,
    28,
    38,
    4,
    22,
    25,
    39,
    43,
    37,
    40,
    35,
    47,
]


def get_tile(sheet, tile_idx, tile_w, tile_h, cols):
    col = tile_idx % cols
    row = tile_idx // cols
    return sheet.crop(
        (col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h)
    )


def rotate_and_trim(tile, angle_deg):
    """Rotates tile in degrees (negative = clockwise) and crops transparent padding."""
    rotated = tile.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
    bbox = rotated.getbbox()
    return rotated.crop(bbox) if bbox else rotated


def save_feature_pair(left_img, right_img, subfolder, prefix, idx, spacing=18):
    wii_dir = os.path.join(OUT_WII, subfolder)
    clip_dir = os.path.join(OUT_CLIP, subfolder)
    os.makedirs(wii_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    canvas_w = left_img.width + right_img.width + spacing + 16
    canvas_h = max(left_img.height, right_img.height) + 16
    pair = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Place Viewer's Left on left, Viewer's Right on right
    y_l = (canvas_h - left_img.height) // 2
    y_r = (canvas_h - right_img.height) // 2
    pair.paste(left_img, (8, y_l), mask=left_img.split()[3])
    pair.paste(
        right_img, (8 + left_img.width + spacing, y_r), mask=right_img.split()[3]
    )

    # 1. Save transparent pair for Wii
    pair.save(os.path.join(wii_dir, f"{prefix}_{idx:02d}.png"))

    # 2. Save composited pair on neutral skin canvas for CLIP embeddings
    canvas = Image.new("RGBA", pair.size, (236, 207, 189, 255))
    canvas.paste(pair, mask=pair.split()[3])
    canvas.convert("RGB").save(os.path.join(clip_dir, f"{prefix}_{idx:02d}.png"))


# ==============================================================================
# 1. FIX EYEBROWS (Arching Upward, Thick Heads at Center, Tails Pointing Out)
# ==============================================================================
if os.path.exists(f"{SRC}/mii_eyebrows.png"):
    brows_sheet = Image.open(f"{SRC}/mii_eyebrows.png").convert("RGBA")

    for b_type in range(24):
        # Canonical eyebrow tile from top rows
        raw_brow = get_tile(brows_sheet, eyebrows_lut[b_type], 36, 54, cols=9)

        # Rotate -90° (clockwise): arch moves to the top, root to the left, tail to the right
        viewer_right_brow = rotate_and_trim(raw_brow, -90)

        # Mirror horizontally to produce the viewer's left eyebrow
        viewer_left_brow = viewer_right_brow.transpose(Image.FLIP_LEFT_RIGHT)

        save_feature_pair(
            viewer_left_brow,
            viewer_right_brow,
            "eyebrows",
            "eyebrow",
            b_type,
            spacing=20,
        )

    print("[+] Eyebrows fixed: 24 horizontal, upward-arching pairs.")

# ==============================================================================
# 2. FIX EYES (Horizontal Lower Lid, Pupils Level, Eyelashes Pointing Out)
# ==============================================================================
if (
    os.path.exists(f"{SRC}/mii_eyes1.png")
    and os.path.exists(f"{SRC}/mii_eyes2.png")
    and os.path.exists(f"{SRC}/mii_eyes3.png")
):
    e1 = Image.open(f"{SRC}/mii_eyes1.png").convert("RGBA")
    e2 = Image.open(f"{SRC}/mii_eyes2.png").convert("RGBA")
    e3 = Image.open(f"{SRC}/mii_eyes3.png").convert("RGBA")

    for eye_type in range(48):
        # Canonical eye sandwich (Whites + Iris + Pupils/Outline)
        tile_idx = eyes_lut[eye_type]
        raw_eye = Image.new("RGBA", (54, 54), (0, 0, 0, 0))
        for layer in [e1, e2, e3]:
            t = get_tile(layer, tile_idx, 54, 54, cols=6)
            raw_eye.paste(t, mask=t.split()[3])

        # Rotate -36° (clockwise): levels the lower eyelid and centers the gaze
        viewer_right_eye = rotate_and_trim(raw_eye, -36)

        # Mirror horizontally to produce the viewer's left eye (eyelashes on outer left)
        viewer_left_eye = viewer_right_eye.transpose(Image.FLIP_LEFT_RIGHT)

        save_feature_pair(
            viewer_left_eye, viewer_right_eye, "eyes", "eye", eye_type, spacing=16
        )

    print("[+] Eyes fixed: 48 horizontal, outward-gazing pairs.")
