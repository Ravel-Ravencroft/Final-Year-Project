import itertools
import numpy as np

from mediapipe.python.solutions import holistic as mp_holistic

from dataclasses import dataclass, field


@dataclass(slots=True)
class HandModel:
    """
    Standard Model for Individual Hand and its Landmarks

    Parameters
    ----------
    landmarks : list[float]
        List of Positions of the Hand

    Attributes
    ----------
    feature_vector : list[np.float64]
        List of length 441 (21*21), containing the angles between all the connections
    """

    landmarks: list[float] = field(repr=False)
    feature_vector: list[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.feature_vector = self._get_feature_vector(self.landmarks)

    @classmethod
    def _get_feature_vector(cls, landmarks: list[float]) -> list[np.float64]:
        """
        Calculates the angles between all the connections

        Parameters
        ----------
        landmarks : list[float]
            List of Positions of the Hand

        Returns
        ----------
        List of Floats
            List of length 441 (21*21), containing the angles between all the connections
        """
        landmarks = np.array(landmarks).reshape((21, 3))
        connections = map(
            lambda t: landmarks[t[1]] - [t[0]],
            mp_holistic.HAND_CONNECTIONS
        )

        angles_list = []
        for connection in itertools.product(connections, repeat=2):
            angle = cls._get_angle_between_connections(connection[0], connection[1])

            angles_list.append(angle if (angle == angle) else 0)

        return angles_list

    @staticmethod
    def _get_angle_between_connections(u: np.ndarray, v: np.ndarray) -> np.float64:
        """
        Calculates the Angle between the Two Connections

        Parameters
        ----------
        u, v : numpy nd-array
            A 3D vector representing a connection

        Returns
        ----------
        Numpy Float64
            The Angle between the Two Connections
        """
        if (np.array_equal(u, v)):
            return np.float64(0)

        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)

        return np.arccos(dot_product / norm)

