import ffmpeg
import json

from pathlib import Path


ROOT_DIR = Path(__file__).parents[1].resolve()
DATASET_DIR = ROOT_DIR / "dataset"
SIGN_DIR = DATASET_DIR / "signs"
VIDEO_DIR = DATASET_DIR / "videos"


def trim(video_id: str, category: str, start_frame: int, end_frame: int) -> str:
	if (not (input_file := VIDEO_DIR / f"{video_id}.mp4").exists()):
		return f"Input File '{input_file.name}' Doesn't Exist!"

	if ((output_file := SIGN_DIR / f"{category}/{category}-{video_id}.mp4").is_file()):
		return f"Output File '{output_file.name}' Already Exists!"

	if (not (output_dir := SIGN_DIR / f"{category}").is_dir()):
		output_dir.mkdir(parents=True, exist_ok=True)

	try:
		(
			ffmpeg
			.input(input_file)
			.trim(start_frame=start_frame, end_frame=end_frame)
			.setpts("PTS-STARTPTS")
			.output(str(output_file))
			.run(quiet=True)
		)
		return f"{input_file.name} -> {output_file.name}"

	except Exception as e:
		return f"An Error Occured while Processing Video [{output_file.name}]!\n{e}"


if (__name__ == "__main__"):
	with open(ROOT_DIR / "sample_list.json", "r") as file:
		sample_list = json.load(file)

	length = len(sample_list)

	for idx, video in enumerate(sample_list):
		print(f"{idx}/{length}:\t{trim(**video)}")

