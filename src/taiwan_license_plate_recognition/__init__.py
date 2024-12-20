import asyncio
import os
from typing import Generator, List

import aiohttp
import cv2
import paddle
from cv2.typing import MatLike
from paddleocr import PaddleOCR
from ultralytics import YOLO

from taiwan_license_plate_recognition.detection import extract_license_plate
from taiwan_license_plate_recognition.recognition import extract_license_number


async def load_detection_model() -> YOLO:
	detection_model_path: str = os.environ.get("DETECTION_MODEL_PATH", "")
	if not detection_model_path:
		raise ValueError("LICENSE NUMBER RECOGNIZER: DETECTION_MODEL_PATH not set.")

	detection_model: YOLO = YOLO(detection_model_path, task="obb")
	print("LICENSE NUMBER RECOGNIZER: detection model loaded.")

	return detection_model


async def load_recognition_model(*args, **kwargs) -> PaddleOCR:
	paddle.disable_signal_handler()

	recognition_model: PaddleOCR = PaddleOCR(*args, **kwargs)
	print("LICENSE NUMBER RECOGNIZER: recognition model loaded.")

	return recognition_model


async def initialize_stream(frame_size: int) -> cv2.VideoCapture:
	stream_source: str = os.environ.get("STREAM_SOURCE", "")
	# stream_source: int = 1
	if not stream_source:
		raise ValueError("LICENSE NUMBER RECOGNIZER: STREAM_SOURCE not set.")

	stream: cv2.VideoCapture = cv2.VideoCapture(stream_source)
	print("LICENSE NUMBER RECOGNIZER: stream initialized.")

	stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
	stream.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size)
	stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size)
	stream.set(cv2.CAP_PROP_FPS, 1)

	return stream


async def get_frame(stream: cv2.VideoCapture, frame_queue: asyncio.Queue) -> None:
	while stream.isOpened():
		response, frame = stream.read()
		if not response:
			await asyncio.sleep(0.5)
			continue

		print("LICENSE NUMBER RECOGNIZER: get frame.")
		await frame_queue.put(frame)


async def detect_license_plate(
	detection_model: YOLO,
	frame_queue: asyncio.Queue,
	frame_size: int,
	inference_device: str,
	image_queue: asyncio.Queue,
) -> None:
	while True:
		frame: MatLike = await frame_queue.get()

		detections: Generator = detection_model.predict(frame, imgsz=frame_size, half=True, device=inference_device)

		await image_queue.put(extract_license_plate(detections, frame_size))


async def recognize_license_number(
	recognition_model: PaddleOCR, image_queue: asyncio.Queue, result_queue: asyncio.Queue
) -> None:
	while True:
		images: List[MatLike] = await image_queue.get()
		if not images:
			continue

		await result_queue.put(list(filter(None, extract_license_number(images, recognition_model))))


async def send_results(result_queue: asyncio.Queue, api_endpoint: str) -> None:
	async with aiohttp.ClientSession() as session:
		while True:
			results: List[str] = await result_queue.get()
			if not results:
				continue

			for result in results:
				print(f"LICENSE NUMBER RECOGNIZER: detect {result}.")
				session.post(api_endpoint, data={"車牌號碼": result, "名稱": "車牌辨識"})
