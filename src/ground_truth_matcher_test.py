from matcher import match_feature, match_color
from PIL import Image
import numpy as np


def run_sanity_tests():
    print("--- 1. Testing Ground-Truth Retrieval ---")
    # Query a hair sprite against the hair gallery
    test_hair = "mii_parts/transparent/hair/hair_00.png"
    matches = match_feature(test_hair, category="hair", top_k=3)
    print(f"Querying {test_hair} -> Top 3 matches:")
    for rank, (name, score) in enumerate(matches, 1):
        print(f"  [{rank}] {name} (Similarity: {score:.4f})")
    assert matches[0][0] == "hair_00", "Top match should be itself!"

    # Query a nose against the nose gallery
    test_nose = "mii_parts/transparent/noses/nose_05.png"
    matches = match_feature(test_nose, category="noses", top_k=3)
    print(f"\nQuerying {test_nose} -> Top 3 matches:")
    for rank, (name, score) in enumerate(matches, 1):
        print(f"  [{rank}] {name} (Similarity: {score:.4f})")
    assert matches[0][0] == "nose_05", "Top match should be itself!"

    print("\nSanity retrieval test passed successfully!")


if __name__ == "__main__":
    run_sanity_tests()
