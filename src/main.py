import time
import cv2
from pathlib import Path

from VideoCapture import VideoCapture
from Segmentator import Segmentator
from utils import mask_to_image, combine_masks, save_image, crop_image_to_mask

from pipeline import process_segmented_face

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data"


def main():
    vid_cap = VideoCapture(source=33)
    vid_cap.start()

    MODEL_PATH = Path(__file__).resolve().parent.parent / "models/face_landmarker.task"
    HAIR_SEG_MODEL_PATH = (
        Path(__file__).resolve().parent.parent / "models/hair_segmenter.tflite"
    )
    segmentator = Segmentator(str(MODEL_PATH), str(HAIR_SEG_MODEL_PATH))

    start_time = time.time()

    cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stream", 1000, 1000)
    cv2.namedWindow("Hair Segmentation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hair Segmentation", 1000, 1000)
    cv2.namedWindow("Face part mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face part mask", 1000, 1000)

    sum_time = 0

    try:
        while True:
            frame = vid_cap.read_queue()
            if frame is None:
                continue

            timestamp = int((time.time() - start_time) * 1000)
            sum_time = sum_time + timestamp

            segmentator.process_frames(frame, timestamp)

            # eye_masks, brow_masks, nose_mask, mouth_mask, face_mask
            annot_frame, combined_mask, masks = segmentator.draw_boundingbox(frame)

            hair_mask = segmentator.get_hair_mask_frame(frame)

            cv2.imshow("Stream", annot_frame)

            if combined_mask is not None:
                cv2.imshow("Face part mask", combined_mask)

            if hair_mask is not None:
                cv2.imshow("Hair Segmentation", hair_mask)

            if (
                masks is not None
                and hair_mask is not None
                and masks[0][0] is not None
                and masks[0][1] is not None
            ):
                key = cv2.waitKey(1) & 0xFF

                if sum_time > 3000:
                    sum_time = 0
                    if len(masks) == 5:
                        save_face_parts(frame, masks, hair_mask)
                elif key == ord("q"):
                    break

    finally:
        vid_cap.stop()
        cv2.destroyAllWindows()


def save_face_parts(frame, masks, hair_mask):

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # save eyes
    eye_mask = combine_masks(masks[0][0], masks[0][1])
    eye_img = mask_to_image(frame, eye_mask)
    eye_img = crop_image_to_mask(eye_img, eye_mask)
    #    save_image(eye_img,OUTPUT_PATH,"eyes")

    # save brows
    brow_mask = combine_masks(masks[1][0], masks[1][1])
    brows_img = mask_to_image(frame, brow_mask)
    brows_img = crop_image_to_mask(brows_img, brow_mask)
    #    save_image(brows_img,OUTPUT_PATH,"brows")

    # save nose
    nose_img = mask_to_image(frame, masks[2])
    nose_img = crop_image_to_mask(nose_img, masks[2])
    #    save_image(nose_img,OUTPUT_PATH,"nose")

    # save mouth
    mouth_img = mask_to_image(frame, masks[3])
    mouth_img = crop_image_to_mask(mouth_img, masks[3])
    #    save_image(mouth_img,OUTPUT_PATH,"mouth")

    # save face
    face_img = mask_to_image(frame, masks[4])
    face_img = crop_image_to_mask(face_img, masks[4])
    #    save_image(face_img,OUTPUT_PATH,"face")

    # save hair
    hair_img = mask_to_image(frame, hair_mask)
    hair_img = crop_image_to_mask(hair_img, hair_mask)
    #    save_image(hair_img,OUTPUT_PATH,"hair")

    print("Attributes saved")

    imgs = {
        "hair": hair_img,
        "eyes": eye_img,
        "eyebrows": brows_img,
        "nose": nose_img,
        "mouth": mouth_img,
        "heads": face_img,
    }

    process_segmented_face(imgs, wii_ip="10.236.53.252", format_type="official")

    pass


if __name__ == "__main__":
    main()
