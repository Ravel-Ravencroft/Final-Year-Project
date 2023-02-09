import cv2
import numpy as np

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import holistic as mp_holistic
from typing import NamedTuple


def mediapipe_detection(image: np.ndarray, model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False
	results = model.process(image)
	return results


def draw_landmarks(image: np.ndarray, results: NamedTuple) ->  None:
	mp_connections = mp_holistic.HAND_CONNECTIONS  # Holistic model

	for landmark_list in (results.left_hand_landmarks, results.right_hand_landmarks):
		mp_drawing.draw_landmarks(
			image=image,
			landmark_list=landmark_list,
			connections=mp_connections,
			landmark_drawing_spec=mp_drawing.DrawingSpec(
				color=(232, 254, 255), thickness=1, circle_radius=4
			),
			connection_drawing_spec=mp_drawing.DrawingSpec(
				color=(255, 249, 161), thickness=2, circle_radius=2
			),
		)

