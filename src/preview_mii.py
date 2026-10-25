"""
Packet-driven Mii preview.

Reads the 74-byte Mii (the same bytes you send to the Wii), decodes every
field with the same bit layout as the C++ RawMiiData struct, then draws the
Mii from the RAW renderMii sprite sheets using the libmii wiki's layer order,
lookup arrays, colour palettes, scale / rotation formulas and anchor points.

Usage:
    render_debug_mii(packet_bytes)            # preferred
    render_debug_mii(matched_results_dict)    # still works (builds the packet)
    python preview_mii.py template.mii out.png
"""

import os
from functools import lru_cache

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Calibration knobs
# --------------------------------------------------------------------------
SHEET_DIR = "renderMii_sprites"  # folder with the raw mii_*.png sheets
FACE_CX = 260  # x of the face centre line (feature coordinate frame)
HEAD_Y = 170  # y of the top of the head tile in the same frame
OUT_SCALE = 3  # upscale factor for the saved PNG
NORMALISE_TINT = True  # brighten grey sprites before multiplying by a palette colour

CANVAS = (520, 440)

# Pivots (pixel inside the tile that scaling/rotation is about).
# Brow pivots were checked against mii_eyebrows.png: the root of tile eyebrows[type] is at its
# bottom-LEFT, so the wiki's left/right anchor labels are swapped. Eye pivots are from the wiki
# (unverified); if the eyes sit too far in/out, swap them.
BROW_PIVOT_VIEWER_RIGHT = (0, 54)  # tile EYEBROWS[type]
BROW_PIVOT_VIEWER_LEFT = (36, 54)  # tile 53 - EYEBROWS[type]
EYE_PIVOT_VIEWER_RIGHT = (36, 36)  # tile EYES[type]
EYE_PIVOT_VIEWER_LEFT = (18, 36)  # tile 95 - EYES[type]

# --------------------------------------------------------------------------
# libmii lookup arrays (type ID -> sheet tile)
# --------------------------------------------------------------------------
HAIRFG = [
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
HAIRBG = [
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
EYEBROWS = [
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
EYES = [
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
NOSES = [5, 0, 2, 3, 7, 6, 4, 10, 8, 9, 1, 11]
LIPS = [
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
LIPS_TINTED = {1, 6, 11, 17, 19}  # only these lip tiles take lipColor

# --------------------------------------------------------------------------
# Palettes
# --------------------------------------------------------------------------
SKIN = [0xECCFBD, 0xF7BC7D, 0xD78A48, 0xF5B189, 0x995122, 0x563010]
HAIR = [0x111111, 0x332222, 0x441111, 0xBB6644, 0x8888AA, 0x443322, 0x996644, 0xDDBB99]
EYE = [0x000000, 0x778887, 0x7E6355, 0x888940, 0x6A84D0, 0x409B5A]
LIP = [0xC76C46, 0xE44E3A, 0xD88789]
GLASSES = [0x626D6C, 0x85703A, 0xAB4E37, 0x426996, 0xB97F27, 0xBDBFB9]


def _col(palette, i):
    v = palette[min(i, len(palette) - 1)]
    return ((v >> 16) & 255, (v >> 8) & 255, v & 255)


# --------------------------------------------------------------------------
# Packet decoding (mirrors the C++ bitfields; bit 0 = LSB, big-endian words)
#   name: (byte offset, byte count, high bit, low bit)
# --------------------------------------------------------------------------
FIELDS = {
    "faceShape": (0x20, 2, 15, 13),
    "skinColor": (0x20, 2, 12, 10),
    "facialFeature": (0x20, 2, 9, 6),
    "hairType": (0x22, 2, 15, 9),
    "hairColor": (0x22, 2, 8, 6),
    "hairPart": (0x22, 2, 5, 5),
    "eyebrowType": (0x24, 4, 31, 27),
    "eyebrowRot": (0x24, 4, 25, 22),
    "eyebrowColor": (0x24, 4, 15, 13),
    "eyebrowSize": (0x24, 4, 12, 9),
    "eyebrowVert": (0x24, 4, 8, 4),
    "eyebrowHoriz": (0x24, 4, 3, 0),
    "eyeType": (0x28, 4, 31, 26),
    "eyeRot": (0x28, 4, 23, 21),
    "eyeVert": (0x28, 4, 20, 16),
    "eyeColor": (0x28, 4, 15, 13),
    "eyeSize": (0x28, 4, 11, 9),
    "eyeHoriz": (0x28, 4, 8, 5),
    "noseType": (0x2C, 2, 15, 12),
    "noseSize": (0x2C, 2, 11, 8),
    "noseVert": (0x2C, 2, 7, 3),
    "lipType": (0x2E, 2, 15, 11),
    "lipColor": (0x2E, 2, 10, 9),
    "lipSize": (0x2E, 2, 8, 5),
    "lipVert": (0x2E, 2, 4, 0),
    "glassesType": (0x30, 2, 15, 12),
    "glassesColor": (0x30, 2, 11, 9),
    "glassesSize": (0x30, 2, 7, 5),
    "glassesVert": (0x30, 2, 4, 0),
    "mustacheType": (0x32, 2, 15, 14),
    "beardType": (0x32, 2, 13, 12),
    "facialHairColor": (0x32, 2, 11, 9),
    "mustacheSize": (0x32, 2, 8, 5),
    "mustacheVert": (0x32, 2, 4, 0),
    "moleOn": (0x34, 2, 15, 15),
    "moleSize": (0x34, 2, 14, 11),
    "moleVert": (0x34, 2, 10, 6),
    "moleHoriz": (0x34, 2, 5, 1),
}


def parse_mii(b: bytes) -> dict:
    if len(b) != 74:
        raise ValueError(f"Mii packet must be 74 bytes, got {len(b)}")
    return {
        k: (int.from_bytes(b[o : o + n], "big") >> lo) & ((1 << (hi - lo + 1)) - 1)
        for k, (o, n, hi, lo) in FIELDS.items()
    }


# --------------------------------------------------------------------------
# Sprite helpers
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _sheet(name):
    p = os.path.join(SHEET_DIR, name)
    return Image.open(p).convert("RGBA") if os.path.exists(p) else None


def _tile(sheet, idx, w, h, cols):
    s = _sheet(sheet)
    if s is None:
        return None
    c, r = idx % cols, idx // cols
    return s.crop((c * w, r * h, (c + 1) * w, (r + 1) * h))


def _tint(img, rgb):
    """Multiply a (grey) sprite by a palette colour, like GRRLIB's tint."""
    if img is None:
        return None
    a = np.asarray(img, dtype=np.float32)
    col = a[..., :3]
    if NORMALISE_TINT:
        m = a[..., 3] > 200
        if m.any():
            p95 = np.percentile(col[m].mean(axis=1), 95)
            col = np.clip(col * min(255.0 / max(p95, 1.0), 1.6), 0, 255)
    col = col * (np.array(rgb, dtype=np.float32) / 255.0)
    out = np.dstack([col, a[..., 3:4]]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _place(cv, sprite, pivot, pos, scale=1.0, angle_cw=0.0, flip=False):
    """Draw `sprite` so its `pivot` pixel lands on `pos`, scaling and rotating
    (clockwise degrees) about that pivot. Returns the new canvas."""
    if sprite is None:
        return cv
    w, h = sprite.size
    px, py = pivot
    if flip:
        sprite = sprite.transpose(Image.FLIP_LEFT_RIGHT)
        px = w - px
    r = int(np.ceil(np.hypot(max(px, w - px), max(py, h - py))))
    pad = Image.new("RGBA", (2 * r, 2 * r), (0, 0, 0, 0))
    pad.paste(sprite, (r - int(px), r - int(py)))  # pivot is now the centre
    if scale != 1.0:
        n = max(1, round(2 * r * scale))
        pad = pad.resize((n, n), Image.LANCZOS)
    if angle_cw:
        pad = pad.rotate(-angle_cw, resample=Image.BICUBIC)  # PIL is anticlockwise
    layer = Image.new("RGBA", cv.size, (0, 0, 0, 0))
    layer.paste(pad, (round(pos[0] - pad.width / 2), round(pos[1] - pad.height / 2)))
    return Image.alpha_composite(cv, layer)


def _eye_tile(idx, iris_rgb):
    """Whites (eyes1) + tinted iris (eyes2) + outline/pupil (eyes3)."""
    out = Image.new("RGBA", (54, 54), (0, 0, 0, 0))
    for sheet, tint in (
        ("mii_eyes1.png", None),
        ("mii_eyes2.png", iris_rgb),
        ("mii_eyes3.png", None),
    ):
        t = _tile(sheet, idx, 54, 54, 6)
        if t is None:
            continue
        out = Image.alpha_composite(out, _tint(t, tint) if tint else t)
    return out


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------
def render_debug_mii(packet, output_path="debug_preview.png"):
    if isinstance(packet, dict):  # old call style: matcher results
        from mii_payload import build_official_74byte_mii

        packet = build_official_74byte_mii(packet)
    m = parse_mii(bytes(packet))

    needed = [
        "mii_heads.png",
        "mii_hairs1.png",
        "mii_hairs2.png",
        "mii_eyes1.png",
        "mii_eyes2.png",
        "mii_eyes3.png",
        "mii_eyebrows.png",
        "mii_noses.png",
        "mii_lips.png",
    ]
    missing = [n for n in needed if _sheet(n) is None]
    if missing:
        print(f"[-] Preview: missing sheets in '{SHEET_DIR}': {missing}")

    cv = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    cx = FACE_CX
    fs = m["faceShape"]
    skin = _col(SKIN, m["skinColor"])
    hair = _col(HAIR, m["hairColor"])
    fhair = _col(HAIR, m["facialHairColor"])

    # Hair lookup: foreground tile, then its matching background tile
    fg = HAIRFG[min(m["hairType"], 71)]
    bg = HAIRBG[fg]
    flip = bool(m["hairPart"])

    # 1. Hair background (behind head)
    if bg != 56:
        cv = _place(
            cv,
            _tint(_tile("mii_hairs2.png", bg, 120, 120, 8), hair),
            (60, 0),
            (cx, HEAD_Y + 10),
            flip=flip,
        )

    # 2. Head
    cv = _place(
        cv, _tint(_tile("mii_heads.png", fs, 120, 120, 4), skin), (60, 0), (cx, HEAD_Y)
    )

    # 3. Facial feature (wrinkles, makeup, ...)
    ff = m["facialFeature"]
    feature_tile = {
        1: fs,
        2: fs,
        3: 33,
        4: 34,
        5: 35,
        6: 36,
        7: 40 + fs,
        8: 38,
        9: 8 + fs,
        10: 16 + fs,
        11: 24 + fs,
    }
    if ff in feature_tile:
        cv = _place(
            cv,
            _tint(_tile("mii_features.png", feature_tile[ff], 120, 120, 8), skin),
            (60, 0),
            (cx, HEAD_Y),
        )

    # 4. Beard
    if m["beardType"]:
        t = _tile("mii_beards.png", (m["beardType"] - 1) * 8 + fs, 120, 140, 8)
        cv = _place(cv, _tint(t, fhair), (60, 0), (cx, HEAD_Y))

    # 5. Mole
    if m["moleOn"]:
        cv = _place(
            cv,
            _sheet("mii_mole.png"),
            (6, 6),
            (cx - 32 + 4 * m["moleHoriz"], 212 + 2.8 * m["moleVert"]),
            scale=0.2 + 0.1 * m["moleSize"],
        )

    # 6. Eyes (Mii's left = viewer's right, drawn first)
    et = EYES[min(m["eyeType"], 47)]
    iris = _col(EYE, m["eyeColor"])
    ex = 9 + 2.6 * m["eyeHoriz"]
    ey = 184 + 2.8 * m["eyeVert"]
    es = 0.3 + 0.1 * m["eyeSize"]
    ea = 11.25 * (7 - m["eyeRot"])
    cv = _place(cv, _eye_tile(et, iris), EYE_PIVOT_VIEWER_RIGHT, (cx + ex, ey), es, +ea)
    cv = _place(
        cv, _eye_tile(95 - et, iris), EYE_PIVOT_VIEWER_LEFT, (cx - ex, ey), es, -ea
    )

    # 7. Eyebrows
    bt = EYEBROWS[min(m["eyebrowType"], 23)]
    bcol = _col(HAIR, m["eyebrowColor"])
    bx = 16 + 2.6 * m["eyebrowHoriz"]
    by = 166 + 2.8 * m["eyebrowVert"]
    bs = 0.3 + 0.1 * m["eyebrowSize"]
    ba = 11.25 * (11 - m["eyebrowRot"])
    cv = _place(
        cv,
        _tint(_tile("mii_eyebrows.png", bt, 36, 54, 9), bcol),
        BROW_PIVOT_VIEWER_RIGHT,
        (cx + bx, by),
        bs,
        +ba,
    )
    cv = _place(
        cv,
        _tint(_tile("mii_eyebrows.png", 53 - bt, 36, 54, 9), bcol),
        BROW_PIVOT_VIEWER_LEFT,
        (cx - bx, by),
        bs,
        -ba,
    )

    # 8. Mouth
    lt = LIPS[min(m["lipType"], 23)]
    lip = _tile("mii_lips.png", lt, 60, 60, 5)
    if lt in LIPS_TINTED:
        lip = _tint(lip, _col(LIP, m["lipColor"]))
    cv = _place(
        cv, lip, (30, 30), (cx, 220 + 2.6 * m["lipVert"]), 0.2 + 0.1 * m["lipSize"]
    )

    # 9. Mustache
    if m["mustacheType"]:
        t = _tile("mii_mustache.png", m["mustacheType"] - 1, 60, 60, 3)
        cv = _place(
            cv,
            _tint(t, fhair),
            (30, 10),
            (cx, 244 + 2.9 * m["mustacheVert"]),
            0.2 + 0.1 * m["mustacheSize"],
        )

    # 10. Nose
    nt = NOSES[min(m["noseType"], 11)]
    nose = _tint(_tile("mii_noses.png", nt, 50, 50, 6), skin)
    cv = _place(
        cv, nose, (25, 30), (cx, 220 + 2.6 * m["noseVert"]), 0.2 + 0.1 * m["noseSize"]
    )

    # 11. Hair foreground (on top of head)
    fg_img = (
        _tile("mii_hairs1.png", fg, 120, 120, 8)
        if fg < 56
        else _tile("mii_hairs2.png", fg - 56, 120, 120, 8)
    )
    cv = _place(cv, _tint(fg_img, hair), (60, 0), (cx, HEAD_Y - 10), flip=flip)

    # 12. Glasses (type >= 6 = sunglasses: untinted frame + tinted lens)
    gt = m["glassesType"]
    if gt:
        gcol = _col(GLASSES, m["glassesColor"])
        gpos = (cx, 195 + 2.6 * m["glassesVert"])
        gscale = 0.1 + 0.1 * m["glassesSize"]
        if gt < 6:
            cv = _place(
                cv,
                _tint(_tile("mii_glasses.png", gt - 1, 180, 72, 4), gcol),
                (90, 32),
                gpos,
                gscale,
            )
        else:
            cv = _place(
                cv,
                _tint(_tile("mii_glasses.png", gt + 2, 180, 72, 4), gcol),
                (90, 32),
                gpos,
                gscale,
            )
            cv = _place(
                cv, _tile("mii_glasses.png", gt - 1, 180, 72, 4), (90, 32), gpos, gscale
            )

    # Crop to content, put on white, upscale, save
    bbox = cv.getbbox()
    if bbox:
        pad = 12
        cv = cv.crop(
            (
                max(bbox[0] - pad, 0),
                max(bbox[1] - pad, 0),
                min(bbox[2] + pad, cv.width),
                min(bbox[3] + pad, cv.height),
            )
        )
    bgimg = Image.new("RGBA", cv.size, (255, 255, 255, 255))
    out = Image.alpha_composite(bgimg, cv).convert("RGB")
    if OUT_SCALE != 1:
        out = out.resize((out.width * OUT_SCALE, out.height * OUT_SCALE), Image.LANCZOS)
    out.save(output_path)

    summary = (
        f"face={fs} skin={m['skinColor']} hair={m['hairType']}/{m['hairColor']} "
        f"brow={m['eyebrowType']}/{m['eyebrowColor']} eye={m['eyeType']}/{m['eyeColor']} "
        f"nose={m['noseType']} lip={m['lipType']}/{m['lipColor']} feat={ff} "
        f"glasses={gt} must={m['mustacheType']} beard={m['beardType']}"
    )
    print(f"[+] Preview rendered from packet -> {output_path}\n    {summary}")
    return out


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "template.mii"
    dst = sys.argv[2] if len(sys.argv) > 2 else "debug_preview.png"
    with open(src, "rb") as f:
        render_debug_mii(f.read(74), dst)
