from typing import Generator, List

import cv2
from PIL import Image

from taiwan_license_plate_recognition.PostProcess import crop_image
from taiwan_license_plate_recognition.Preprocess import affine_transform


def extract_license_plate(results: Generator) -> List[Image.Image]:
	license_plates: List[Image.Image] = []

	for result in results:
		for obb in result.obb.xyxyxyxy:
			cropped_image = crop_image(result.orig_img, obb)
			cropped_image, _ = affine_transform(cropped_image, 384)
			cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
			license_plates.append(cropped_image)

	return license_plates
