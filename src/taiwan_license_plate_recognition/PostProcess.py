import re
from string import punctuation
from typing import List

import cv2
import torch
from cv2.typing import MatLike


def crop_image(image: MatLike, bounded_box: torch.Tensor) -> MatLike:
	x, y, w, h = cv2.boundingRect(bounded_box.numpy())

	return image[y : y + h, x : x + w]


def remove_non_alphanum(s: str) -> str:
	return s.translate(str.maketrans("", "", punctuation)).replace(" ", "")


def filter_license_number(candidates: List[str]) -> str:
	candidates_without_hyphen: List[str] = [remove_non_alphanum(candidate) for candidate in candidates]

	filter_candidates: List[str] = []
	for i, (c, cwh) in enumerate(zip(candidates, candidates_without_hyphen)):
		if (
			re.match(r"^[A-Z\d]{2}-[A-Z\d]{4}$", c) is None
			and re.match(r"^[A-Z\d]{4}-[A-Z\d]{2}$", c) is None
			and re.match(r"^[A-Z\d]{3}-[A-Z\d]{3}$", c) is None
			and re.match(r"^[A-Z\d]{3}-[A-Z\d]{4}$", c) is None
		):
			continue

		filter_candidates.append(cwh.replace("-", ""))

	return filter_candidates[0] if len(set(filter_candidates)) >= 1 else ""
