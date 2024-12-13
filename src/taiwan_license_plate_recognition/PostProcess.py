import re
from string import punctuation
from typing import Optional

import cv2
import numpy
from cv2.typing import MatLike


def crop_image(image: MatLike, bounded_box: numpy.ndarray) -> MatLike:
	x, y, w, h = cv2.boundingRect(bounded_box)

	return image[y : y + h, x : x + w]


def remove_non_alphanum(s: str) -> Optional[str]:
	return s.translate(str.maketrans("", "", punctuation)).replace(" ", "")


def validate_license_number(candidate: str) -> Optional[str]:
	if (
		re.match(r"^[A-Z\d]{2}-[A-Z\d]{4}$", candidate) is None
		and re.match(r"^[A-Z\d]{4}-[A-Z\d]{2}$", candidate) is None
		and re.match(r"^[A-Z\d]{3}-[A-Z\d]{3}$", candidate) is None
		and re.match(r"^[A-Z\d]{3}-[A-Z\d]{4}$", candidate) is None
	):
		return None

	return candidate.replace("-", "")
