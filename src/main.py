import os
from asyncio.tasks import Task
from typing import List

import cv2
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from ultralytics import YOLO

from datasets.utils.file_utils import asyncio
from taiwan_license_plate_recognition import (
	detect_license_plate,
	get_frame,
	initialize_stream,
	load_detection_model,
	load_recognition_model,
	recognize_license_number,
	send_results,
)


async def main() -> None:
	load_dotenv()

	inference_device: str = os.environ.get("INFERENCE_DEVICE", "cpu")
	frame_size: int = int(os.environ.get("FRAME_SIZE", 640))
	api_endpoint: str = os.environ.get("API_ENDPOINT", "")

	detection_model: YOLO = load_detection_model()

	recognition_model: PaddleOCR = load_recognition_model(
		lang="en",
		binarize=True,
		use_angle_cls=True,
		max_text_length=8,
		use_space_char=False,
		device=inference_device,
	)

	stream: cv2.VideoCapture = initialize_stream(frame_size)

	frame_queue: asyncio.Queue = asyncio.Queue()
	image_queue: asyncio.Queue = asyncio.Queue()
	result_queue: asyncio.Queue = asyncio.Queue()

	tasks: List[Task] = [
		asyncio.create_task(get_frame(stream, frame_queue)),
		asyncio.create_task(
			detect_license_plate(detection_model, frame_queue, frame_size, inference_device, image_queue)
		),
		asyncio.create_task(recognize_license_number(recognition_model, image_queue, result_queue)),
		asyncio.create_task(send_results(result_queue, api_endpoint)),
	]

	await asyncio.gather(*tasks, return_exceptions=True)


asyncio.run(main())
