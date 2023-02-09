import cv2
import logging
import pandas as pd
import pyvirtualcam

from mediapipe.python.solutions.holistic import Holistic
from pathlib import Path

from utils import mediapipe_utils
from utils.recorder_utils import SignRecorder
from utils.webcam_utils import WebcamManager


ROOT_FOLDER = Path(__file__).parents[1].resolve()


def passthrough() -> None:
    reference_signs: pd.DataFrame = pd.read_pickle(ROOT_FOLDER / "data/dataset.pkl")

    logging.info(msg=f"Sign Count: {reference_signs[['name']].groupby(['name']).value_counts()}")

    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(reference_signs=reference_signs, seq_len=30)

    # Grabbing the Cam dimensions
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    logging.info(msg=f"Webcam Width: {width}, Height: {height}, FPS: {fps}")


    # Set up the PyVirtualCam and Mediapipe environments
    with (
        pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam,
        Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic
    ):
        logging.info(msg=f"Using Virtual Camera: {cam.device}")

        while cap.isOpened():
            # Read feed
            _, frame = cap.read()

            # Make detections
            results = mediapipe_utils.mediapipe_detection(frame, holistic)

            # Process results
            detected_sign, _ = sign_recorder.process_results(results)

            # Update the frame (draw landmarks & display result)
            WebcamManager.update(frame, results, detected_sign)

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cam.send(frame)
            cam.sleep_until_next_frame()

            # TODO: Rework to listen to Console Keystroke and Continuous Recording
            # pressedKey = cv2.waitKey(1) & 0xFF
            # if pressedKey == ord("r"):  # Record pressing r
            #     sign_recorder.record()
            # elif pressedKey == ord("q"):  # Break pressing q
            #     break

    cap.release()

