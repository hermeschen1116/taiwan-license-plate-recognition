from typing import List

import numpy
from PIL import Image
from cv2.typing import MatLike

from taiwan_license_plate_recognition.PostProcess import crop_image, filter_license_number, remove_non_alphanum
from taiwan_license_plate_recognition.Preprocess import add_letterbox


def extract_license_plate(result, size: int = 640) -> List[MatLike]:
	license_plates: List[MatLike] = []

	for obb in result.obb.xyxyxyxy:
		cropped_image = crop_image(result.orig_img, obb.numpy())
		cropped_image, _ = add_letterbox(cropped_image, size)
		license_plates.append(cropped_image)

	return license_plates


def extract_license_number_paddleocr(images: List[MatLike], model=None) -> List[str]:
	image_arrays = [numpy.asarray(image, dtype=numpy.uint8) for image in images]

	results = [model.ocr(image)[0] for image in image_arrays]

	return [
		filter_license_number([(result[i][1][0]) for i in range(len(result))]) if result is not None else ""
		for result in results
	]


def extract_license_number_trocr(
	images: List[Image.Image], model=None, processor=None, max_length: int = 64
) -> List[str]:
	if len(images) == 0:
		return []

	encode_image = processor(images, return_tensors="pt").pixel_values

	generated_ids = model.generate(encode_image, max_length=max_length)

	results = processor.batch_decode(generated_ids, skip_special_tokens=True)

	return [remove_non_alphanum(result) for result in results]
