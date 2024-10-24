import os

import cv2
import numpy
from cv2.typing import MatLike
from dotenv import load_dotenv
from matplotlib import pyplot
from typing_extensions import Tuple


def to_grayscale(image: MatLike) -> MatLike:
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def affine_transform(image: MatLike, new_size: int) -> Tuple[MatLike, numpy.ndarray]:
	image_shape: numpy.ndarray = numpy.array(image.shape[:2])

	scale_ratio: numpy.floating = numpy.min(new_size / image_shape)
	translation_array: numpy.ndarray = (image_shape * scale_ratio - new_size) * -0.5

	affine_array: numpy.ndarray = numpy.array(
		[[scale_ratio, 0, translation_array[1]], [0, scale_ratio, translation_array[0]]], dtype=numpy.float32
	)

	return cv2.warpAffine(
		image,
		affine_array,
		dsize=[new_size, new_size],
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=(114, 114, 114),
	), cv2.invertAffineTransform(affine_array)


def to_RGB(image: MatLike) -> MatLike:
	return image.transpose(2, 0, 1)[..., numpy.newaxis]


def add_letterbox(image: MatLike, new_size: int) -> Tuple[MatLike, numpy.ndarray]:
	transformed_image, reverse_affine_array = affine_transform(image, new_size)

	return to_RGB(transformed_image), reverse_affine_array


if __name__ == "__main__":
	load_dotenv()

	image: MatLike = cv2.imread(f"{os.environ.get("PROJECT_ROOT", "")}/test_image.png") / 255
	pyplot.subplot(1, 3, 1)
	pyplot.title("original")
	pyplot.imshow(image[..., ::-1])

	affine_image, reverse_affine_array = affine_transform(image, 640)
	pyplot.subplot(1, 3, 2)
	pyplot.title("affine_transform")
	pyplot.imshow(affine_image[..., ::-1])

	reverse_affine_image = cv2.warpAffine(affine_image, reverse_affine_array, image.shape[:2][::-1])
	pyplot.subplot(1, 3, 3)
	pyplot.title("reverse")
	pyplot.imshow(reverse_affine_image[..., ::-1])

	pyplot.show()
