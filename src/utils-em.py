import numpy as np
from PIL import Image


def pad_to_square(img: Image.Image, bg_color: int = 255) -> Image.Image:
    """Pads a non-square PIL image into a square canvas centered on neutral gray."""
    w, h = img.size
    if w == h:
        return img
    max_side = max(w, h)
    total_size = int(max_side * 1.1)  # 10% breathing room

    # Always pad as RGB on neutral gray
    mode = "RGB"
    fill = (bg_color, bg_color, bg_color)

    # If the input still has alpha, pad with transparent border first
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        mode = "RGBA"
        fill = (bg_color, bg_color, bg_color, 0)

    new_img = Image.new(mode, (total_size, total_size), fill)
    offset_x = (total_size - w) // 2
    offset_y = (total_size - h) // 2
    new_img.paste(img, (offset_x, offset_y))
    return new_img


def standardize_crop_for_clip(crop_input, bg_color: int = 255) -> Image.Image:
    """
    Guarantees exact identical processing whether indexing Mii sprites
    or querying real cropped inputs:
      1. Load as PIL
      2. Alpha-composite transparent regions onto solid neutral gray (128)
      3. Pad to square canvas (128)
      4. Convert to 3-channel grayscale [L, L, L] for CLIP
    """
    if isinstance(crop_input, str):
        img = Image.open(crop_input)
    elif isinstance(crop_input, np.ndarray):
        img = Image.fromarray(crop_input)
    else:
        img = crop_input

    # 1. Handle transparency if present (Mii sprites, segmented PNGs)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        np_img = np.array(img)
        alpha = np_img[:, :, 3] / 255.0
        rgb = np_img[:, :, :3]

        bg = np.full_like(rgb, bg_color)
        composite = (rgb * alpha[:, :, None] + bg * (1.0 - alpha[:, :, None])).astype(
            np.uint8
        )
        img = Image.fromarray(composite)
    else:
        img = img.convert("RGB")

    # 2. Pad to square on the same gray color
    img = pad_to_square(img, bg_color=bg_color)

    # 3. Convert to 3-channel grayscale for CLIP
    return img.convert("L").convert("RGB")
