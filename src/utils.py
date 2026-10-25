import cv2
import numpy as np
from PIL import Image


def combine_masks(mask1, mask2):
    combined_mask = cv2.bitwise_or(mask1, mask2)
    return combined_mask


"""
def mask_to_image(frame,mask):
    masked_image = cv2.bitwise_and(frame,frame,mask=mask)

    #set white background
    masked_image = np.where(masked_image[..., None] > 0, frame, 255)
    return masked_image
"""


def mask_to_image(frame, mask):
    # convert mask to 2d
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_expanded = (mask > 0)[..., np.newaxis]

    # apply mask
    result = np.where(mask_expanded, frame, 255)

    return result.astype(np.uint8)


def crop_image_to_mask(image, mask):
    y_inds, x_inds = np.where(mask > 0)

    if len(y_inds) == 0 or len(x_inds) == 0:
        return image

    y_min, y_max = np.min(y_inds), np.max(y_inds)
    x_min, x_max = np.min(x_inds), np.max(x_inds)

    pad = 10
    h, w, _ = image.shape
    y_min = max(0, y_min - pad)
    y_max = min(h, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad)

    # Crop the image to these dimensions
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def save_image(image, filepath, name):
    cv2.imwrite(str(filepath) + f"/{name}.png", image)


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


def standardize_crop_for_clip(
    crop_input, bg_color: int = 255, enhance_texture: bool = False
) -> Image.Image:
    if isinstance(crop_input, str):
        img = Image.open(crop_input)
    elif isinstance(crop_input, np.ndarray):
        img = Image.fromarray(crop_input)
    else:
        img = crop_input

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

    img = pad_to_square(img, bg_color=bg_color)
    gray = np.array(img.convert("L"))

    # Enhance internal hair/facial texture on real camera crops
    if enhance_texture:
        non_bg = gray < 250
        if np.any(non_bg):
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            # Blend enhanced texture with original grayscale inside the feature mask
            gray = np.where(
                non_bg, cv2.addWeighted(enhanced, 0.7, gray, 0.3, 0), 255
            ).astype(np.uint8)

    return Image.fromarray(gray).convert("RGB")
