import cv2
import numpy as np

from utils import mediapipe_utils


HEIGHT = 600


class WebcamManager:
    """
    Object that displays the Webcam output, draws the landmarks detected, and outputs the predicted Sign.
    """
    @classmethod
    def update(cls, frame: np.ndarray, results, detected_sign: str) -> np.ndarray:
        # Draw landmarks
        mediapipe_utils.draw_landmarks(image=frame, results=results)

        WIDTH = int(HEIGHT * len(frame[0]) / len(frame))
        # Resize frame
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        # Flip the image vertically for mirror effect
        frame = cv2.flip(frame, 1)

        # Write result if there is
        frame = cls._draw_text(frame=frame, detected_sign=detected_sign)

        return frame

    @classmethod
    def _draw_text(
        cls,
        frame: np.ndarray,
        detected_sign: str,
        font=cv2.FONT_HERSHEY_COMPLEX,
        font_size=1,
        font_thickness=2,
        offset=int(HEIGHT * 0.02),
        bg_color=(245, 242, 176, 0.85)
    ) -> np.ndarray:

        window_w = int(HEIGHT * len(frame[0]) / len(frame))

        (text_w, text_h), _ = cv2.getTextSize(text=detected_sign, fontFace=font, fontScale=font_size, thickness=font_thickness)

        text_x, text_y = int((window_w - text_w) / 2), HEIGHT - text_h - offset

        cv2.rectangle(img=frame, pt1=(0, text_y - offset), pt2=(window_w, HEIGHT), color=bg_color, thickness=-1)

        cv2.putText(
            frame,
            detected_sign,
            (text_x, text_y + text_h + font_size - 1),
            font,
            font_size,
            (118, 62, 37),
            font_thickness,
        )

        return frame

