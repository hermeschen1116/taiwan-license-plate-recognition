from typing import List

import cv2
import numpy
from PIL import Image

from taiwan_license_plate_recognition.PostProcess import crop_image, filter_license_number, remove_non_alphanum
from taiwan_license_plate_recognition.Preprocess import affine_transform


def extract_license_plate(result) -> List[Image.Image]:
	license_plates: List[Image.Image] = []

	for obb in result.obb.xyxyxyxy:
		cropped_image = crop_image(result.orig_img, obb)
		cropped_image, _ = affine_transform(cropped_image, 384)
		cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
		license_plates.append(cropped_image)

	return license_plates


def extract_license_number_paddleocr(images: List[Image.Image], model=None) -> List[str]:
	results = [model.ocr(numpy.asarray(image, dtype=numpy.uint8))[0] for image in images]

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
