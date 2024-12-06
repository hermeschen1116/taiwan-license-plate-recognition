from string import punctuation

import cv2
import torch
from cv2.typing import MatLike


def crop_image(image: MatLike, bounded_box: torch.Tensor) -> MatLike:
	x, y, w, h = cv2.boundingRect(bounded_box.numpy())

	return image[y : y + h, x : x + w]


def remove_non_alphanum(s: str) -> str:
	return s.strip(punctuation)
