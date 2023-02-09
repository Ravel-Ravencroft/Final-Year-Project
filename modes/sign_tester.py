import cv2
import logging
import pandas as pd

from mediapipe.python.solutions.holistic import Holistic
from pathlib import Path

from utils import mediapipe_utils
from utils.recorder_utils import SignRecorder
from utils.webcam_utils import WebcamManager


ROOT_FOLDER = Path(__file__).parents[1].resolve()

RED = (25, 35, 240)
WHITE = (245, 242, 226)


def sign_tester() -> None:
	reference_signs: pd.DataFrame = pd.read_pickle(ROOT_FOLDER / "data/dataset.pkl")

	logging.info(msg=f"Sign Count: {reference_signs[['name']].groupby(['name']).value_counts()}")

	# Object that stores mediapipe results and computes sign similarities
	sign_recorder = SignRecorder(reference_signs=reference_signs, seq_len=30)

	# Turn on the webcam
	cap = cv2.VideoCapture(0)

	# Set up the Mediapipe environment
	with Holistic(
		min_detection_confidence=0.5, min_tracking_confidence=0.5
	) as holistic:
		while cap.isOpened():
			# Read feed
			_, frame = cap.read()

			# Make detections
			results = mediapipe_utils.mediapipe_detection(frame, holistic)

			# Process results
			detected_sign, is_recording = sign_recorder.process_results(results)

			# Choose Circle Colour
			color = RED if is_recording else WHITE

			# Update the frame
			cv2.circle(frame, (30, 30), 20, color, -1)

			# Update the frame (draw landmarks & display result)
			frame = WebcamManager.update(frame, results, detected_sign)

			# Display the Image
			cv2.imshow("Sign Tester", frame)

			pressedKey = cv2.waitKey(1) & 0xFF
			if pressedKey == ord("r"):  # Record pressing r
				sign_recorder.record()
			elif pressedKey == ord("q"):  # Break pressing q
				break

		cv2.destroyAllWindows()

	cap.release()

