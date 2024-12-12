import itertools
from typing import List

from PIL import Image
from cv2.typing import MatLike

from taiwan_license_plate_recognition.PostProcess import crop_image, filter_license_number, remove_non_alphanum
from taiwan_license_plate_recognition.Preprocess import add_letterbox


def extract_license_plate(results, size: int = 640) -> List[MatLike]:
	license_plates: List[MatLike] = []

	for result in results:
		for obb in result.obb.xyxyxyxy:
			cropped_image = crop_image(result.orig_img, obb.numpy())
			cropped_image, _ = add_letterbox(cropped_image, size)
			license_plates.append(cropped_image)

	return license_plates


def extract_license_number_paddleocr(images: List[MatLike], model=None) -> List[str]:
	predictions = [model.ocr(image) for image in images]

	results = [result[0][1][0] for result in filter(None, itertools.chain.from_iterable(predictions))]

	return [filter_license_number(result) for result in results]


def extract_license_number_trocr(
	images: List[Image.Image], model=None, processor=None, max_length: int = 64
) -> List[str]:
	if len(images) == 0:
		return []

	encode_image = processor(images, return_tensors="pt").pixel_values

	generated_ids = model.generate(encode_image, max_length=max_length)

	results = processor.batch_decode(generated_ids, skip_special_tokens=True)

	return [remove_non_alphanum(result) for result in results]
