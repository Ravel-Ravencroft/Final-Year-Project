import numpy as np

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from typing import NamedTuple


def extract_landmarks(results: NamedTuple):
    """
    Extract the results of both hands and convert them to a np array of size (1, 21 * 3).
    If a hand doesn't appear, return an array of zeros

    Parameters
        results: mediapipe object that contains the 3D position of all keypoints.

    Returns
        Two np arrays of size (1, 21 * 3), (1, nb_keypoints * nb_coordinates), corresponding to both hands.
    """
    left_hand = np.zeros(63).tolist()
    if results.left_hand_landmarks:
        left_hand = _landmark_to_array(results.left_hand_landmarks).reshape(63).tolist()

    right_hand = np.zeros(63).tolist()
    if results.right_hand_landmarks:
        right_hand = _landmark_to_array(results.right_hand_landmarks).reshape(63).tolist()

    return left_hand, right_hand


def _landmark_to_array(mp_landmark_list: NormalizedLandmarkList):
    """
    Return a np array of size (nb_keypoints * 3)
    """
    return np.nan_to_num([[landmark.x, landmark.y, landmark.z] for landmark in mp_landmark_list.landmark])

