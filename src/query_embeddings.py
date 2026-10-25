import sys
import cv2
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.jit._serialization"
)
import clip
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup Models & Device
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load the offline Mii hair embeddings
INDEX_PATH = "mii_hair_embeddings.pt"
index_data = torch.load(INDEX_PATH, map_location=device)
mii_embeddings = index_data["embeddings"].to(device)  # shape: [72, 512]
mii_filenames = index_data["filenames"]

# Setup MediaPipe Segmenter
base_options = python.BaseOptions(model_asset_path="selfie_multiclass.tflite")
options = vision.ImageSegmenterOptions(
    base_options=base_options, output_category_mask=True
)
segmenter = vision.ImageSegmenter.create_from_options(options)

# Canonical Mii Hair Color Palette (from libmii)
MII_HAIR_COLORS_HEX = [
    0x111111,
    0x332222,
    0x441111,
    0xBB6644,
    0x8888AA,
    0x443322,
    0x996644,
    0xDDBB99,
]


def hex_to_rgb(hex_val):
    return np.array(
        [(hex_val >> 16) & 0xFF, (hex_val >> 8) & 0xFF, hex_val & 0xFF], dtype=np.uint8
    )


# Convert all 8 Mii colors to CIELAB once upfront
MII_LAB_COLORS = [
    cv2.cvtColor(hex_to_rgb(h).reshape(1, 1, 3), cv2.COLOR_RGB2LAB)
    .reshape(3)
    .astype(np.float32)
    for h in MII_HAIR_COLORS_HEX
]


def get_hair_mask_and_crop(image_rgb: np.ndarray):
    """Runs MediaPipe segmenter and returns binary mask (hair=1) and cropped silhouette."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = segmenter.segment(mp_image)
    category_mask = result.category_mask.numpy_view()

    # Hair is category 1 in selfie_multiclass
    hair_mask = (category_mask == 1).astype(np.uint8)
    if not np.any(hair_mask):
        return None, None

    # White silhouette on black canvas (matches offline Mii index)
    silhouette = (hair_mask * 255).astype(np.uint8)
    return hair_mask, Image.fromarray(silhouette).convert("RGB")


def extract_best_color(image_rgb: np.ndarray, hair_mask: np.ndarray) -> int:
    """Calculates median RGB of hair pixels and finds nearest Mii color in CIELAB."""
    hair_pixels = image_rgb[hair_mask == 1]

    # Compute median RGB (robust against lighting glare / extreme shadow)
    median_rgb = np.median(hair_pixels, axis=0).astype(np.uint8)

    # Convert sampled human hair color to CIELAB
    human_lab = (
        cv2.cvtColor(median_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)
        .reshape(3)
        .astype(np.float32)
    )

    # Euclidean distance in LAB space (Delta E)
    distances = [np.linalg.norm(human_lab - mii_lab) for mii_lab in MII_LAB_COLORS]
    return int(np.argmin(distances))


def query_mii(image_path: str):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1. Extract Mask & Binary Silhouette
    hair_mask, silhouette_pil = get_hair_mask_and_crop(image_rgb)
    if silhouette_pil is None:
        print("No hair detected in image.")
        return

    # 2. Geometry Branch (CLIP)
    query_tensor = clip_preprocess(silhouette_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embed = clip_model.encode_image(query_tensor)
        query_embed /= query_embed.norm(dim=-1, keepdim=True)

        # Dot product with pre-normalized gallery
        similarities = (query_embed @ mii_embeddings.T).squeeze(0)
        best_match_idx = torch.argmax(similarities).item()

    matched_hair_filename = mii_filenames[best_match_idx]

    # 3. Color Branch (CIELAB)
    best_color_id = extract_best_color(image_rgb, hair_mask)

    print("\n--- Match Results ---")
    print(
        f"Matched Hair Asset : {matched_hair_filename} (Similarity: {similarities[best_match_idx]:.3f})"
    )
    print(
        f"Matched Color ID   : {best_color_id} (Hex: #{MII_HAIR_COLORS_HEX[best_color_id]:06X})"
    )

    return matched_hair_filename, best_color_id


if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else "test_face.jpg"
    query_mii(test_img)
