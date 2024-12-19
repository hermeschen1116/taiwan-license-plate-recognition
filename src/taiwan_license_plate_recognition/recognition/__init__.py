import itertools
from typing import List, Optional

from cv2.typing import MatLike
from paddleocr import PaddleOCR

from taiwan_license_plate_recognition.recognition.PostProcess import validate_license_number


def extract_license_number(images: List[MatLike], model: PaddleOCR) -> List[Optional[str]]:
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
