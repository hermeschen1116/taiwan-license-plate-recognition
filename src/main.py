import os
from typing import List, Optional

import cv2
from cv2.typing import MatLike
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
from taiwan_license_plate_recognition.Utils import get_num_of_workers


async def main() -> None:
	load_dotenv()

	inference_device: str = os.environ.get("INFERENCE_DEVICE", "cpu")
	num_workers: int = int(os.environ.get("NUM_WORKERS", get_num_of_workers()))
	frame_size: int = int(os.environ.get("FRAME_SIZE", 640))
	api_endpoint: str = os.environ.get("API_ENDPOINT", "")

	detection_model: YOLO = await load_detection_model()

	recognition_model: PaddleOCR = await load_recognition_model(
		lang="en",
		binarize=True,
		use_angle_cls=True,
		max_text_length=8,
		use_space_char=False,
		device=inference_device,
		use_mp=True,
		total_process_num=num_workers,
	)

	stream: cv2.VideoCapture = await initialize_stream(frame_size)

	while stream.isOpened():
		key: int = cv2.waitKey(90)
		if key == ord('q') or key == 27:
			stream.release()
			break

		frame: Optional[MatLike] = await get_frame(stream)
		if frame is None:
			continue

		cropped_images: List[MatLike] = await detect_license_plate(detection_model, frame, frame_size, inference_device)
		license_numbers: List[str] = await recognize_license_number(recognition_model, cropped_images)

		await send_results(license_numbers, api_endpoint)

asyncio.run(main())
