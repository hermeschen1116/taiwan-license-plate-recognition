import cv2
import numpy
import torch
from cv2.typing import MatLike


def crop_image(image: MatLike, bounded_box: torch.Tensor) -> MatLike:
	mask: numpy.ndarray = numpy.zeros_like(image)
	cv2.fillPoly(mask, [bounded_box.to(torch.int).numpy()], [255, 255, 255])

	cropped_image: MatLike = cv2.bitwise_and(image, mask)

	return cropped_image
