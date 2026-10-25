import os
from PIL import Image

SRC = "renderMii_sprites"
OUT_WII = "mii_parts/transparent"  # Clean alpha PNGs for rendering
OUT_CLIP = "mii_parts/clip_preview"  # Neutral skin canvas for CLIP vector embeddings

os.makedirs(OUT_WII, exist_ok=True)
os.makedirs(OUT_CLIP, exist_ok=True)

# --- OFFICIAL LIBMII TRANSLATION ARRAYS ---
hairfg = [
    59,
    42,
    65,
    49,
    40,
    44,
    52,
    47,
    45,
    63,
    51,
    54,
    36,
    37,
    48,
    70,
    61,
    56,
    64,
    43,
    53,
    58,
    50,
    27,
    69,
    41,
    39,
    46,
    66,
    71,
    33,
    11,
    12,
    0,
    35,
    57,
    30,
    14,
    25,
    4,
    1,
    31,
    26,
    24,
    3,
    6,
    62,
    13,
    15,
    7,
    19,
    2,
    17,
    67,
    29,
    20,
    9,
    34,
    18,
    8,
    22,
    60,
    23,
    55,
    21,
    32,
    16,
    28,
    10,
    38,
    5,
    68,
]
hairbg = [
    56,
    56,
    56,
    56,
    56,
    56,
    56,
    56,
    56,
    56,
    56,
    56,
    16,
    56,
    56,
    56,
    56,
    56,
    17,
    18,
    56,
    19,
    20,
    56,
    56,
    56,
    21,
    56,
    56,
    56,
    56,
    56,
    56,
    56,
    56,
    56,
    22,
    23,
    56,
    56,
    24,
    25,
    56,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    56,
    56,
    48,
    49,
    50,
    51,
    52,
    53,
    56,
]
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
noses_lut = [5, 0, 2, 3, 7, 6, 4, 10, 8, 9, 1, 11]
lips_lut = [
    6,
    1,
    14,
    16,
    17,
    5,
    10,
    12,
    7,
    13,
    8,
    19,
    23,
    11,
    22,
    18,
    9,
    15,
    21,
    2,
    20,
    3,
    4,
    0,
]


def get_tile(sheet, tile_idx, tile_w, tile_h, cols):
    col = tile_idx % cols
    row = tile_idx // cols
    x = col * tile_w
    y = row * tile_h
    return sheet.crop((x, y, x + tile_w, y + tile_h))


def save_asset(img, subfolder, filename):
    wii_dir = os.path.join(OUT_WII, subfolder)
    clip_dir = os.path.join(OUT_CLIP, subfolder)
    os.makedirs(wii_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    # 1. Pure transparent RGBA for Wii
    img.save(os.path.join(wii_dir, filename))

    # 2. Composited on neutral skin tone for CLIP embeddings (RGB)
    canvas = Image.new("RGBA", img.size, (236, 207, 189, 255))  # skincol[0] = ECCFBD
    canvas.paste(img, mask=img.split()[3])
    canvas.convert("RGB").save(os.path.join(clip_dir, filename))


# ==============================================================================
# 1. GLASSES (720x216 -> 180x72 tiles, 4 cols x 3 rows = 11 tiles)
# ==============================================================================
if os.path.exists(f"{SRC}/mii_glasses.png"):
    glasses_sheet = Image.open(f"{SRC}/mii_glasses.png").convert("RGBA")
    for g in range(8):  # 8 canonical glasses styles
        tile = get_tile(glasses_sheet, g, 180, 72, cols=4)
        save_asset(tile, "glasses", f"glasses_{g:02d}.png")
    print("[+] Extracted 8 glasses correctly (180x72).")

# ==============================================================================
# 2. EYES (Compositing eyes1 + eyes2 + eyes3 into complete eye pairs)
# ==============================================================================
if (
    os.path.exists(f"{SRC}/mii_eyes1.png")
    and os.path.exists(f"{SRC}/mii_eyes2.png")
    and os.path.exists(f"{SRC}/mii_eyes3.png")
):
    e1 = Image.open(f"{SRC}/mii_eyes1.png").convert("RGBA")  # Whites
    e2 = Image.open(f"{SRC}/mii_eyes2.png").convert("RGBA")  # Iris
    e3 = Image.open(f"{SRC}/mii_eyes3.png").convert("RGBA")  # Outline / Pupils

    for eye_type in range(48):
        left_idx = eyes_lut[eye_type]
        right_idx = 95 - eyes_lut[eye_type]

        # Composite Left Eye (e1 + e2 + e3)
        left_eye = Image.new("RGBA", (54, 54), (0, 0, 0, 0))
        for layer in [e1, e2, e3]:
            t = get_tile(layer, left_idx, 54, 54, cols=6)
            left_eye.paste(t, mask=t.split()[3])

        # Composite Right Eye (e1 + e2 + e3)
        right_eye = Image.new("RGBA", (54, 54), (0, 0, 0, 0))
        for layer in [e1, e2, e3]:
            t = get_tile(layer, right_idx, 54, 54, cols=6)
            right_eye.paste(t, mask=t.split()[3])

        # Combine both eyes horizontally into a pair for easy CLIP feature matching (108x54)
        eye_pair = Image.new("RGBA", (108, 54), (0, 0, 0, 0))
        eye_pair.paste(left_eye, (0, 0))
        eye_pair.paste(right_eye, (54, 0))

        save_asset(eye_pair, "eyes", f"eye_{eye_type:02d}.png")
        save_asset(left_eye, "eyes_single", f"eye_left_{eye_type:02d}.png")
    print("[+] Extracted & composited 48 complete 3-layer eye types.")

# ==============================================================================
# 3. EYEBROWS (324x324 -> 36x54 tiles, 9 cols x 6 rows)
# ==============================================================================
if os.path.exists(f"{SRC}/mii_eyebrows.png"):
    brows_sheet = Image.open(f"{SRC}/mii_eyebrows.png").convert("RGBA")
    for b_type in range(24):
        left_idx = eyebrows_lut[b_type]
        right_idx = 53 - eyebrows_lut[b_type]

        left_brow = get_tile(brows_sheet, left_idx, 36, 54, cols=9)
        right_brow = get_tile(brows_sheet, right_idx, 36, 54, cols=9)

        brow_pair = Image.new("RGBA", (72, 54), (0, 0, 0, 0))
        brow_pair.paste(left_brow, (0, 0))
        brow_pair.paste(right_brow, (36, 0))

        save_asset(brow_pair, "eyebrows", f"eyebrow_{b_type:02d}.png")
    print("[+] Extracted 24 eyebrows (36x54).")

# ==============================================================================
# 4. HEADS (480x240 -> 120x120 tiles, 4 cols x 2 rows = 8 shapes)
# ==============================================================================
if os.path.exists(f"{SRC}/mii_heads.png"):
    heads_sheet = Image.open(f"{SRC}/mii_heads.png").convert("RGBA")
    for h in range(8):
        tile = get_tile(heads_sheet, h, 120, 120, cols=4)
        save_asset(tile, "heads", f"head_{h:02d}.png")
    print("[+] Extracted 8 head shapes (120x120).")

# ==============================================================================
# 5. MOUTHS / LIPS (300x300 -> 60x60 tiles, 5 cols x 5 rows = 24 types)
# ==============================================================================
if os.path.exists(f"{SRC}/mii_lips.png"):
    lips_sheet = Image.open(f"{SRC}/mii_lips.png").convert("RGBA")
    for l_type in range(24):
        tile = get_tile(lips_sheet, lips_lut[l_type], 60, 60, cols=5)
        save_asset(tile, "mouths", f"mouth_{l_type:02d}.png")
    print("[+] Extracted 24 mouth types (60x60).")

# ==============================================================================
# 6. NOSES (300x100 -> 50x50 tiles, 6 cols x 2 rows = 12 types)
# ==============================================================================
if os.path.exists(f"{SRC}/mii_noses.png"):
    noses_sheet = Image.open(f"{SRC}/mii_noses.png").convert("RGBA")
    for n_type in range(12):
        tile = get_tile(noses_sheet, noses_lut[n_type], 50, 50, cols=6)
        save_asset(tile, "noses", f"nose_{n_type:02d}.png")
    print("[+] Extracted 12 noses (50x50).")

# ==============================================================================
# 7. HAIR (mii_hairs1.png & mii_hairs2.png -> 960x840, 120x120 tiles, 8 cols x 7 rows)
# ==============================================================================
if os.path.exists(f"{SRC}/mii_hairs1.png") and os.path.exists(f"{SRC}/mii_hairs2.png"):
    h1 = Image.open(f"{SRC}/mii_hairs1.png").convert("RGBA")
    h2 = Image.open(f"{SRC}/mii_hairs2.png").convert("RGBA")

    for h_type in range(72):
        fg_idx = hairfg[h_type]
        bg_idx = hairbg[fg_idx]

        # Extract Foreground Hair
        if fg_idx < 56:
            fg_tile = get_tile(h1, fg_idx, 120, 120, cols=8)
        else:
            fg_tile = get_tile(h2, fg_idx - 56, 120, 120, cols=8)

        # Composite with Background Hair if present (bg_idx != 56 is not blank)
        hair_full = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        if bg_idx != 56:
            bg_tile = get_tile(h2, bg_idx, 120, 120, cols=8)
            hair_full.paste(bg_tile, (0, 0), mask=bg_tile.split()[3])
        hair_full.paste(fg_tile, (0, 0), mask=fg_tile.split()[3])

        save_asset(hair_full, "hair", f"hair_{h_type:02d}.png")
    print("[+] Extracted all 72 hairstyles composited with foreground & background.")

# ==============================================================================
# 8. MUSTACHES & BEARDS
# ==============================================================================
if os.path.exists(f"{SRC}/mii_mustache.png"):
    mus = Image.open(f"{SRC}/mii_mustache.png").convert("RGBA")
    for m in range(3):
        tile = get_tile(mus, m, 60, 60, cols=3)
        save_asset(tile, "mustaches", f"mustache_{m:02d}.png")
    print("[+] Extracted 3 mustaches.")

if os.path.exists(f"{SRC}/mii_beards.png"):
    beards = Image.open(f"{SRC}/mii_beards.png").convert("RGBA")
    # 24 tiles: 8 cols x 3 rows (120x140)
    for b in range(3):
        # Sample for face shape 0
        tile = get_tile(beards, b * 8, 120, 140, cols=8)
        save_asset(tile, "beards", f"beard_{b:02d}.png")
    print("[+] Extracted 3 beards.")

print("\nDone! All parts correctly parsed per the Libmii doc specification.")
