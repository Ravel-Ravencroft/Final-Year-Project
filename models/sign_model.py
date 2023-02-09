import numpy as np

from dataclasses import dataclass, field

from models.hand_model import HandModel


@dataclass(slots=True)
class SignModel:
    """
    Parameters
        x_hand_list: List of all landmarks for each frame of a video

    Attributes
        has_x_hand: bool; True if x hand is detected in the video, otherwise False
        xh_embedding: ndarray; Array of shape (n_frame, nb_connections * nb_connections)
    """
    left_hand_list: list[list[float]] = field(repr=False)
    right_hand_list: list[list[float]] = field(repr=False)
    has_left_hand: bool = field(init=False)
    has_right_hand: bool = field(init=False)
    lh_embedding: list[list[float]] = field(init=False)
    rh_embedding: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.has_left_hand = np.sum(self.left_hand_list) != 0
        self.has_right_hand = np.sum(self.right_hand_list) != 0

        self.lh_embedding = self._get_embedding_from_landmark_list(self.left_hand_list)
        self.rh_embedding = self._get_embedding_from_landmark_list(self.right_hand_list)

    @staticmethod
    def _get_embedding_from_landmark_list(hand_list: list[list[float]]) -> list[list[float]]:
        """
        Parameters
            hand_list: List of all landmarks for each frame of a video

        Return
            Array of shape (n_frame, nb_connections * nb_connections) containing the 
            feature_vectors of the hand for each frame
        """
        embedding = []

        for frame_idx in range(len(hand_list)):
            if np.sum(hand_list[frame_idx]) == 0:
                continue

            hand_gesture = HandModel(hand_list[frame_idx])
            embedding.append(hand_gesture.feature_vector)

        return embedding

