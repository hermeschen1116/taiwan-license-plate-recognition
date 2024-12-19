import cv2
import numpy
from cv2.typing import MatLike


def crop_image(image: MatLike, bounded_box: numpy.ndarray) -> MatLike:
	x, y, w, h = cv2.boundingRect(bounded_box)

	return image[y : y + h, x : x + w]
