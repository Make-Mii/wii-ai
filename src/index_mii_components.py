import os
import glob
import torch
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.jit._serialization"
)
import clip
from PIL import Image
import numpy as np

from utils import standardize_crop_for_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Point to whichever folder has all your category subdirectories
BASE_DIR = "./mii_parts/transparent"
OUTPUT_FILE = "mii_galleries.pt"

CATEGORIES = [
    "hair",
    "eyes",
    "eyebrows",
    "noses",
    "mouths",
    "heads",
    "glasses",
    "beards",
    "mustaches",
]

database = {}

for cat in CATEGORIES:
    cat_dir = os.path.join(BASE_DIR, cat)
    if not os.path.isdir(cat_dir):
        continue

    paths = sorted(glob.glob(os.path.join(cat_dir, "*.png")))
    if not paths:
        continue

    tensors = [preprocess(standardize_crop_for_clip(p)) for p in paths]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        embeddings = model.encode_image(batch)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)

    database[cat] = {
        "embeddings": embeddings.cpu(),
        "filenames": [os.path.splitext(os.path.basename(p))[0] for p in paths],
    }
    print(f"Indexed {len(paths):2d} items for category '{cat}'")

torch.save(database, OUTPUT_FILE)
print(f"\nAll feature galleries compiled into '{OUTPUT_FILE}'.")
