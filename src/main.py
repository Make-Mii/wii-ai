import time
import threading
import cv2
import numpy as np
from pathlib import Path

from VideoCapture import VideoCapture
from Segmentator import Segmentator
from utils import mask_to_image, combine_masks, crop_image_to_mask, save_image,cartoonify
from pipeline import process_segmented_face

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data"
WII_IP = "10.236.53.252"




def main():
    vid_cap = VideoCapture(source=0, max_queue_size=2)
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
 

    # Background worker state so CLIP never blocks the video stream
    worker_busy = threading.Event()

    def async_worker(frame_bgr, masks_copy, hair_mask_copy,save_bool):
        try:
            save_face_parts(frame_bgr, masks_copy, hair_mask_copy,save_bool)
        except Exception as e:
            print(f"[Worker Error] {e}")
        finally:
            worker_busy.clear()

    try:
        while True:
            frame = vid_cap.read_queue()
            if frame is None:
                continue


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


            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            save_key_pressed = (key == ord("s"))

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
                and ((now - last_trigger_time > 3.0) or key == ord(" ") or save_key_pressed)
            ):
                last_trigger_time = now
                worker_busy.set()
                threading.Thread(
                    target=async_worker,
                    args=(frame.copy(), masks, hair_mask.copy(),save_key_pressed),
                    daemon=True,
                ).start()

    finally:
        vid_cap.stop()
        cv2.destroyAllWindows()


def save_face_parts(frame_bgr, masks, hair_mask,save_images=False):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Convert BGR camera frame to RGB so CIELAB color matching gets true colors!
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 1. Eyes
    eye_mask = combine_masks(masks[0][0], masks[0][1])
    eye_img = crop_image_to_mask(mask_to_image(frame_rgb, eye_mask), eye_mask)
    eye_img = cartoonify(eye_img)

    # 2. Eyebrows
    brow_mask = combine_masks(masks[1][0], masks[1][1])
    brows_img = crop_image_to_mask(mask_to_image(frame_rgb, brow_mask), brow_mask)
    brows_img = cartoonify(brows_img)
    # 3. Nose
    nose_img = crop_image_to_mask(mask_to_image(frame_rgb, masks[2]), masks[2])
    nose_img = cartoonify(nose_img)

    # 4. Mouth
    mouth_img = crop_image_to_mask(mask_to_image(frame_rgb, masks[3]), masks[3])
    mouth_img = cartoonify(mouth_img)

    # 5. Face / Skin (exclude eyes, brows, and mouth from skin color sampling!)
    skin_only_mask = masks[4].copy()
    skin_only_mask[eye_mask > 0] = 0
    skin_only_mask[brow_mask > 0] = 0
    skin_only_mask[masks[3] > 0] = 0
    face_img = crop_image_to_mask(mask_to_image(frame_rgb, masks[4]), masks[4])
    face_img = cartoonify(face_img)

    # 6. Hair
    hair_img = crop_image_to_mask(mask_to_image(frame_rgb, hair_mask), hair_mask)
    hair_img = cartoonify(hair_img)

    imgs = {
        "hair": hair_img,
        "eyes": eye_img,
        "eyebrows": brows_img,
        "nose": nose_img,
        "mouth": mouth_img,
        "heads": face_img,
        "glasses": True,
    }

    if save_images:
        for name, img in imgs.items():
            if img is not None and img.size > 0:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #img_bgr = cartoonify(img_bgr)
                save_image(img_bgr, OUTPUT_PATH, name)

    process_segmented_face(imgs, wii_ip=WII_IP, format_type="official")


if __name__ == "__main__":
    main()
