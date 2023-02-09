import json

from pathlib import Path

from modes import display as Webcam
from modes import sign_tester as Screen


ROOT_FOLDER = Path(__file__).parent.resolve()

with open(ROOT_FOLDER / "config.json") as file:
    CONFIG = json.load(file)


if (__name__ == "__main__"):
    if (CONFIG["production_mode"]):
        Webcam.passthrough()
    else:
        Screen.sign_tester()

