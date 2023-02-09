import fastdtw as fdtw
import numpy as np
import pandas as pd

from models.sign_model import SignModel


def dtw_distances(recorded_sign: SignModel, reference_signs: pd.DataFrame):
	"""
	Use DTW to compute similarity between the recorded sign & the reference signs

	Parameters
		recorded_sign: a SignModel object containing the data gathered during recording
		reference_signs: a Pandas DataFrame containing the reference signs

	Returns
		Return a sign dictionary sorted by the distances from the recorded sign
	"""
	reference_signs["distance"] = reference_signs.apply(lambda row: _compute_distances(recorded_sign=recorded_sign, reference_sign=row["sign_model"]), axis="columns")

	return reference_signs.sort_values(by=["distance"])


def _compute_distances(recorded_sign: SignModel, reference_sign: SignModel) -> np.float64:
	# Checks if both Signs have the Same Number of Hands, Returns Infinity if Not
	if (recorded_sign.has_left_hand != reference_sign.has_left_hand) or (
		recorded_sign.has_right_hand != reference_sign.has_right_hand
	):
		return np.inf

	distance = 0

	if recorded_sign.has_left_hand:
		distance += fdtw.fastdtw(recorded_sign.lh_embedding, reference_sign.lh_embedding)[0]
	if recorded_sign.has_right_hand:
		distance += fdtw.fastdtw(recorded_sign.rh_embedding, reference_sign.rh_embedding)[0]

	return distance

