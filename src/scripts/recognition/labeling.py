import os
from typing import Any, Dict, List

import cv2
import datasets
import numpy
import paddle
from cv2.typing import MatLike
from datasets import load_dataset
from dotenv import load_dotenv
from paddleocr import PaddleOCR

from taiwan_license_plate_recognition.Utils import get_num_of_workers
from taiwan_license_plate_recognition.recognition.PostProcess import remove_non_alphanum

paddle.disable_signal_handler()
load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()

dataset = load_dataset("hermeschen1116/taiwan-license-plate-ocr", num_proc=num_workers)

dataset = dataset.cast_column("image", datasets.Image(decode=True))


def encode_image(image) -> numpy.ndarray:
	cv2_image = cv2.cvtColor(numpy.asarray(image, dtype=numpy.uint8), cv2.COLOR_RGB2BGR)

	return cv2_image


dataset.set_transform(encode_image, columns=["image"], output_all_columns=True)
dataset.set_format("numpy", columns=["image"], output_all_columns=True)

reader = PaddleOCR(
	lang="en",
	use_angle_cls=True,
	max_text_length=8,
	total_process_num=num_workers,
	use_mp=True,
	use_space_char=False,
	binarize=True,
)


def is_labeled(annotations) -> bool:
	return len(annotations) != 0 and any([annotation["transcription"] != "###" for annotation in annotations])


def get_annotation(image: MatLike, label: str) -> List[Dict[str, Any]]:
	try:
		detections = reader.ocr(image)[0]
	except IndexError:
		return []

	if detections is None:
		return []

	results: List[Dict[str, Any]] = [{"transcription": result[1][0], "points": result[0]} for result in detections]

	for result in results:
		if result["transcription"] != label and result["transcription"] != remove_non_alphanum(label):
			result["transcription"] = "###"
		else:
			result["transcription"] = label

	if not is_labeled(results):
		return []

	return results


dataset = dataset.map(lambda sample: {"annotation": get_annotation(sample["image"], sample["label"])})

dataset = dataset.filter(lambda sample: len(sample) != 0, input_columns=["annotation"])

dataset.push_to_hub("taiwan-license-plate-ocr", commit_message="add label", embed_external_files=True)
