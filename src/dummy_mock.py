from pipeline import process_segmented_face

if __name__ == "__main__":
    from PIL import Image

    print("--- Testing Pipeline with Mock Crop Dictionary ---")
    mock_crops = {
        "hair": Image.open("mii_parts/transparent/hair/hair_00.png"),
        "eyes": Image.open("mii_parts/transparent/eyes/eye_03.png"),
        "eyebrows": Image.open("mii_parts/transparent/eyebrows/eyebrow_02.png"),
        "noses": Image.open("mii_parts/transparent/noses/nose_05.png"),
        "mouths": Image.open("mii_parts/transparent/mouths/mouth_00.png"),
        "heads": Image.open("mii_parts/transparent/heads/head_00.png"),
    }

    # Generate payload (omit wii_ip or set to your laptop's localhost "127.0.0.1" to test UDP)
    final_mii = process_segmented_face(
        mock_crops, wii_ip="192.168.137.39", format_type="official"
    )

    print("\nGenerated Mii Payload:")
    for k, v in list(final_mii.items())[:10]:
        print(f"  {k:20s}: {v}")
    print(f"  ... ({len(final_mii)} fields total)")
