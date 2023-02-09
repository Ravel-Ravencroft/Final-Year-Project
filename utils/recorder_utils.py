import numpy as np
import pandas as pd

from models.sign_model import SignModel
from utils import dtw_utils, landmark_utils


class SignRecorder:
    def __init__(self, reference_signs: pd.DataFrame, seq_len: int=50):
        self.is_recording: bool = False
        self.seq_len: int = seq_len
        self.recorded_results: list = []
        self.reference_signs: pd.DataFrame = reference_signs

    def record(self) -> None:
        """
        Initialize sign_distances & start recording
        """
        self.reference_signs["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results) -> tuple[str, bool]:
        """
        If the SignRecorder is in the recording state, it stores the landmarks during seq_len
        frames and then computes the sign distances

        Paramaters
            results: mediapipe output

        Returns
            The word predicted (blank text if there is no distances) & the recording state.
        """
        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self.compute_distances()
                print(self.reference_signs.head(5)[["name", "video_name", "distance"]][self.reference_signs["distance"] != np.inf])

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording

        return self._get_sign_predicted(), self.is_recording

    def compute_distances(self):
        """
        Updates the distance column of the reference_signs and resets recording variables.
        """
        left_hand_list, right_hand_list = [], []
        for results in self.recorded_results:
            left_hand, right_hand = landmark_utils.extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(left_hand_list, right_hand_list)

        # Compute sign similarity with DTW (ascending order)
        self.reference_signs = dtw_utils.dtw_distances(recorded_sign, self.reference_signs)

        # Reset variables
        self.recorded_results.clear()
        self.is_recording = False

    def _get_sign_predicted(self, batch_size=5, threshold=0.6) -> str:
        """
        Method that outputs the sign that appears the most in the list of closest reference 
        signs, only if its proportion within the batch is greater than the threshold.

        Parameters
            batch_size: Size of the batch of reference signs that will be compared to the recorded sign

            threshold: If the proportion of the most represented sign in the batch is greater than 
            the threshold, we output the sign_name. If not, we output "Sign not found".

        Returns
            The name of the predicted sign.
        """
        # Get the list (of size batch_size) of the most similar reference signs
        most_common = self.reference_signs.iloc[:batch_size][["name"]].groupby(["name"]).value_counts()

        if ((most_common.max() / batch_size) < threshold):
            return "Sign Not Recognised!"

        return most_common.idxmax().title()

