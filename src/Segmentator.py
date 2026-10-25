import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


import sys
from collections import deque
from pathlib import Path
import cv2


from typing import List
import numpy as np


class Segmentator:
    r_eye = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
        247,
        30,
        29,
        27,
        28,
        56,
        190,
    ]
    l_eye = [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
        414,
        286,
        258,
        257,
        259,
        260,
        467,
    ]
    r_eyebrow = [107, 55, 221, 222, 223, 224, 225, 113, 124, 156, 71, 63, 105, 66]
    l_eyebrow = [336, 286, 441, 442, 443, 444, 445, 342, 353, 383, 301, 293, 334, 296]
    nose = [
        168,
        122,
        188,
        174,
        217,
        209,
        49,
        64,
        98,
        240,
        97,
        141,
        2,
        370,
        326,
        305,
        327,
        294,
        358,
        279,
        429,
        355,
        437,
        343,
        412,
        351,
    ]
    mouth = [
        0,
        267,
        269,
        270,
        409,
        287,
        375,
        321,
        405,
        314,
        17,
        84,
        181,
        91,
        146,
        57,
        61,
        185,
        40,
        39,
        37,
    ]
    face_outline = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]
    key_landmarks = r_eye + l_eye + r_eyebrow + l_eyebrow + nose + mouth + face_outline

    def __init__(self, model_path, hair_seg_model_path):

        self.model_path = model_path
        self.hair_seg_model_path = hair_seg_model_path

        # FACE LANDMARK DETECTOR
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.latest_landmarks = None
        self.latest_hair_seg = None

        options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            result_callback=self.print_result,
        )

        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )

        # HAIR SEGMENTER
        self.HairBaseOptions = mp.tasks.BaseOptions
        self.HairImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        self.HairVisionRunningMode = mp.tasks.vision.RunningMode

        options = self.HairImageSegmenterOptions(
            base_options=self.HairBaseOptions(
                model_asset_path=self.hair_seg_model_path
            ),
            running_mode=self.HairVisionRunningMode.LIVE_STREAM,
            output_category_mask=True,
            result_callback=self.print_hair_result,
        )

        self.HairImageSegmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
            options
        )

    # callback for hair segmentation
    def print_hair_result(
        self, result: List[mp.Image], output_image: mp.Image, timestamp_ms: int
    ):
        self.latest_hair_seg = result

    # callback function
    def print_result(self, result, output_image: mp.Image, timestamp_ms: int):
        # print('face landmarker result: {}'.format(result))
        self.latest_landmarks = result

    def draw_boundingbox(self, frame):
        if self.latest_landmarks is None:
            return frame, None, None  # ,None,None,None,None

        height, width, _ = frame.shape
        annot_frame = frame.copy()
        combined_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        all_masks = []

        # masks for each face part
        r_eye_m = self.create_mask_from_indices_brows(frame, self.r_eye)
        l_eye_m = self.create_mask_from_indices_brows(frame, self.l_eye)

        if r_eye_m is not None and l_eye_m is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11))
            r_eye_m = cv2.dilate(r_eye_m, kernel, iterations=1)
            l_eye_m = cv2.dilate(l_eye_m, kernel, iterations=1)

        eye_masks = [r_eye_m, l_eye_m]
        all_masks.append(eye_masks)

        r_brow_m = self.create_mask_from_indices_brows(frame, self.r_eyebrow)
        l_brow_m = self.create_mask_from_indices_brows(frame, self.l_eyebrow)

        brow_masks = [r_brow_m, l_brow_m]
        all_masks.append(brow_masks)

        nose_m = self.create_mask_from_indices(frame, self.nose)
        mouth_m = self.create_mask_from_indices(frame, self.mouth)
        face_outline_m = self.create_mask_from_indices(frame, self.face_outline)

        all_masks.append(nose_m)
        all_masks.append(mouth_m)
        all_masks.append(face_outline_m)

        # unique colours
        if face_outline_m is not None:
            combined_canvas[face_outline_m > 0] = [50, 50, 50]  # Dark Grey outline
        if r_brow_m is not None and l_brow_m is not None:
            combined_canvas[(r_brow_m > 0) | (l_brow_m > 0)] = [
                0,
                255,
                255,
            ]  # Yellow Eyebrows
        if r_eye_m is not None and l_eye_m is not None:
            combined_canvas[(r_eye_m > 0) | (l_eye_m > 0)] = [255, 0, 0]  # Blue Eyes
        if nose_m is not None:
            combined_canvas[nose_m > 0] = [0, 255, 0]  # Green Nose
        if mouth_m is not None:
            combined_canvas[mouth_m > 0] = [0, 0, 255]

        for face_landmarks in self.latest_landmarks.face_landmarks:
            for idx, landmark in enumerate(face_landmarks):
                if idx in self.key_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    # Draw a tiny red dot (radius=1, BGR color=(0, 0, 255)) for each landmark
                    cv2.circle(annot_frame, (x, y), 1, (0, 0, 255), -1)

                    # cv2.putText(frame,text=str(idx),org=(x,y),fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.5,color=(0,0,225),thickness=1,lineType=cv2.LINE_AA)

        return (
            annot_frame,
            combined_canvas,
            all_masks,
        )  # ,brow_masks,nose_m,mouth_m,face_outline_m

    def get_hair_mask_frame(self, frame):
        if self.latest_hair_seg is None:
            return None

        mask_image = self.latest_hair_seg.category_mask.numpy_view()

        # Category 1 is hair in selfie_multiclass / hair_segmenter
        hair_mask = (mask_image == 1).astype("uint8") * 255
        if not np.any(hair_mask):
            hair_mask = (mask_image > 0).astype("uint8") * 255

        hair_mask = cv2.resize(
            hair_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return hair_mask

    def create_mask_from_indices(self, frame, landmark_indices):
        if self.latest_landmarks is None:
            return None

        height, width, _ = frame.shape

        mask = np.zeros((height, width), dtype=np.uint8)

        for face_landmarks in self.latest_landmarks.face_landmarks:
            pts = []

            for idx in landmark_indices:
                if idx < len(face_landmarks):
                    landmark = face_landmarks[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    pts.append((x, y))

            if pts:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)

                # 2. Sort the points counter-clockwise (or clockwise) around the center using arctan2
                pts = sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))

                pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

                cv2.fillPoly(mask, [pts_array], 255)
            return mask

    def create_mask_from_indices_brows(self, frame, landmark_indices):
        if self.latest_landmarks is None:
            return None

        height, width, _ = frame.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        for face_landmarks in self.latest_landmarks.face_landmarks:
            pts = []
            for idx in landmark_indices:
                if idx < len(face_landmarks):
                    landmark = face_landmarks[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    pts.append((x, y))

            if pts:
                pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

                hull = cv2.convexHull(pts_array)
                cv2.fillPoly(mask, [hull], 255)

        return mask

    def process_frames(self, frame, timestamp):
        mp_img = mp.Image(mp.ImageFormat.SRGB, data=frame)

        self.FaceLandmarker.detect_async(mp_img, timestamp)
        self.HairImageSegmenter.segment_async(mp_img, timestamp)
