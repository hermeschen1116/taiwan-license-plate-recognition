import itertools
from typing import List, Optional, Union

from PIL.Image import Image
from cv2.typing import MatLike
from optimum.intel.openvino import OVModelForVision2Seq
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from taiwan_license_plate_recognition.recognition.PostProcess import (
	remove_non_alphanum as remove_non_alphanum,
	validate_license_number,
)


def extract_license_number_paddleocr(images: List[MatLike], model: PaddleOCR) -> List[Optional[str]]:
	if len(images) == 0:
		return []

	try:
		predictions = [model.ocr(image) for image in images]
	except IndexError:
		# deal with Paddle OCR internal error
		return []

	results: List[str] = [
		result[0][1][0] if result is not None else "" for result in itertools.chain.from_iterable(predictions)
	]

	return [validate_license_number(result) for result in results]


def extract_license_number_trocr(
	images: List[Image],
	model: Union[OVModelForVision2Seq, VisionEncoderDecoderModel],
	processor: TrOCRProcessor,
	max_length: int = 8,
) -> List[Optional[str]]:
	if len(images) == 0:
		return []

	encode_image = processor(images, return_tensors="pt").pixel_values

	generated_ids = model.generate(encode_image, max_length=max_length)

	results = processor.batch_decode(generated_ids, skip_special_tokens=True)

	return [validate_license_number(result) for result in results]
