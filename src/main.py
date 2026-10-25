import time
import threading
import cv2
import numpy as np
from pathlib import Path

from VideoCapture import VideoCapture
from Segmentator import Segmentator
from utils import mask_to_image, combine_masks, crop_image_to_mask
from pipeline import process_segmented_face

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data"
WII_IP = "10.236.53.252"


def apply_camera_adjustments(
    frame: np.ndarray, brightness: int, contrast: int, sat: int, gamma_val: int
) -> np.ndarray:
    """Applies real-time brightness, contrast, saturation, and gamma adjustments."""
    # 1. Brightness (-100 to +100) & Contrast (0.5x to 2.0x)
    alpha = contrast / 100.0
    beta = brightness - 100
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # 2. Gamma correction (50 to 200 -> 0.5 to 2.0)
    gamma = max(0.1, gamma_val / 100.0)
    if abs(gamma - 1.0) > 0.05:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            "uint8"
        )
        adjusted = cv2.LUT(adjusted, table)

    # 3. Saturation (0 to 200 -> 0.0x to 2.0x)
    if abs(sat - 100) > 2:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (sat / 100.0), 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return adjusted


def main():
    vid_cap = VideoCapture(source=33, max_queue_size=2)
    vid_cap.start()

    MODEL_PATH = Path(__file__).resolve().parent.parent / "models/face_landmarker.task"
    HAIR_SEG_MODEL_PATH = (
        Path(__file__).resolve().parent.parent / "models/hair_segmenter.tflite"
    )
    segmentator = Segmentator(str(MODEL_PATH), str(HAIR_SEG_MODEL_PATH))

    start_time = time.time()
    last_trigger_time = time.time()

    cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stream", 900, 750)
    cv2.namedWindow("Hair Segmentation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hair Segmentation", 600, 600)
    cv2.namedWindow("Face part mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face part mask", 600, 600)

    # Add on-the-fly camera sliders to the Stream window
    cv2.createTrackbar(
        "Brightness", "Stream", 100, 200, lambda x: None
    )  # 100 = neutral (0)
    cv2.createTrackbar("Contrast", "Stream", 100, 200, lambda x: None)  # 100 = 1.0x
    cv2.createTrackbar("Gamma", "Stream", 100, 200, lambda x: None)  # 100 = 1.0
    cv2.createTrackbar("Saturation", "Stream", 100, 200, lambda x: None)  # 100 = 1.0x

    # Background worker state so CLIP never blocks the video stream
    worker_busy = threading.Event()

    def async_worker(frame_bgr, masks_copy, hair_mask_copy):
        try:
            save_face_parts(frame_bgr, masks_copy, hair_mask_copy)
        except Exception as e:
            print(f"[Worker Error] {e}")
        finally:
            worker_busy.clear()

    try:
        while True:
            raw_frame = vid_cap.read_queue()
            if raw_frame is None:
                continue

            # Apply live slider settings
            b_val = cv2.getTrackbarPos("Brightness", "Stream")
            c_val = cv2.getTrackbarPos("Contrast", "Stream")
            g_val = cv2.getTrackbarPos("Gamma", "Stream")
            s_val = cv2.getTrackbarPos("Saturation", "Stream")
            frame = apply_camera_adjustments(raw_frame, b_val, c_val, s_val, g_val)

            # Initialize before the while loop:
            last_ts = -1

            # Inside the while loop:
            timestamp = int((time.time() - start_time) * 1000)
            if timestamp <= last_ts:
                timestamp = last_ts + 1
            last_ts = timestamp
            segmentator.process_frames(frame, timestamp)

            annot_frame, combined_mask, masks = segmentator.draw_boundingbox(frame)
            hair_mask = segmentator.get_hair_mask_frame(frame)

            cv2.imshow("Stream", annot_frame)
            if combined_mask is not None:
                cv2.imshow("Face part mask", combined_mask)
            if hair_mask is not None:
                cv2.imshow("Hair Segmentation", hair_mask)

            # Show live debug preview if available
            if Path("debug_preview.png").exists():
                prev = cv2.imread("debug_preview.png")
                if prev is not None:
                    cv2.imshow("Mii Preview", prev)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # Trigger every 3 seconds (or immediately on SPACEBAR) if worker is free
            ready = (
                masks is not None
                and len(masks) == 5
                and hair_mask is not None
                and masks[0][0] is not None
                and masks[0][1] is not None
            )

            now = time.time()
            if (
                ready
                and not worker_busy.is_set()
                and ((now - last_trigger_time > 3.0) or key == ord(" "))
            ):
                last_trigger_time = now
                worker_busy.set()
                threading.Thread(
                    target=async_worker,
                    args=(frame.copy(), masks, hair_mask.copy()),
                    daemon=True,
                ).start()

    finally:
        vid_cap.stop()
        cv2.destroyAllWindows()


def save_face_parts(frame_bgr, masks, hair_mask):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Convert BGR camera frame to RGB so CIELAB color matching gets true colors!
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 1. Eyes
    eye_mask = combine_masks(masks[0][0], masks[0][1])
    eye_img = crop_image_to_mask(mask_to_image(frame_rgb, eye_mask), eye_mask)

    # 2. Eyebrows
    brow_mask = combine_masks(masks[1][0], masks[1][1])
    brows_img = crop_image_to_mask(mask_to_image(frame_rgb, brow_mask), brow_mask)

    # 3. Nose
    nose_img = crop_image_to_mask(mask_to_image(frame_rgb, masks[2]), masks[2])

    # 4. Mouth
    mouth_img = crop_image_to_mask(mask_to_image(frame_rgb, masks[3]), masks[3])

    # 5. Face / Skin (exclude eyes, brows, and mouth from skin color sampling!)
    skin_only_mask = masks[4].copy()
    skin_only_mask[eye_mask > 0] = 0
    skin_only_mask[brow_mask > 0] = 0
    skin_only_mask[masks[3] > 0] = 0
    face_img = crop_image_to_mask(mask_to_image(frame_rgb, skin_only_mask), masks[4])

    # 6. Hair
    hair_img = crop_image_to_mask(mask_to_image(frame_rgb, hair_mask), hair_mask)

    imgs = {
        "hair": hair_img,
        "eyes": eye_img,
        "eyebrows": brows_img,
        "nose": nose_img,
        "mouth": mouth_img,
        "heads": face_img,
    }

    process_segmented_face(imgs, wii_ip=WII_IP, format_type="official")


if __name__ == "__main__":
    main()
