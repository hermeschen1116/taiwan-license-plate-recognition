from typing import Generator, List

from cv2.typing import MatLike

from taiwan_license_plate_recognition.detection.PostProcess import crop_image
from taiwan_license_plate_recognition.recognition.Preprocess import add_letterbox


def extract_license_plate(results: Generator, size: int = 640) -> List[MatLike]:
	license_plates: List[MatLike] = []

	for result in results:
		for obb in result.obb.xyxyxyxy:
			cropped_image: MatLike = crop_image(result.orig_img, obb.numpy())
			cropped_image, _ = add_letterbox(cropped_image, size)
			license_plates.append(cropped_image)

	return license_plates
