import asyncio
import os
from typing import Generator, List, Optional

import cv2
import paddle
import requests
from cv2.typing import MatLike
from paddleocr import PaddleOCR
from ultralytics import YOLO

from taiwan_license_plate_recognition.detection import extract_license_plate
from taiwan_license_plate_recognition.recognition import extract_license_number


async def load_detection_model() -> YOLO:
	detection_model_path: str = os.environ.get("DETECTION_MODEL_PATH", "")
	if detection_model_path == "":
		raise ValueError("LICENSE NUMBER RECOGNIZER: DETECTION_MODEL_PATH not set.")

	return YOLO(detection_model_path, task="obb")

async def load_recognition_model(*args, **kwargs) -> PaddleOCR:
	paddle.disable_signal_handler()
	return PaddleOCR(*args, **kwargs)

async def initialize_stream(frame_size: int) -> cv2.VideoCapture:
	# stream_source: str = os.environ.get("STREAM_SOURCE", "")
	stream_source: int = 1
	if stream_source == "":
		raise ValueError("LICENSE NUMBER RECOGNIZER: STREAM_SOURCE not set.")

	stream: cv2.VideoCapture = cv2.VideoCapture(stream_source)
	stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))

	stream.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size)
	stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size)
	stream.set(cv2.CAP_PROP_FPS, 1)

	return stream

async def get_frame(stream: cv2.VideoCapture) -> Optional[MatLike]:
	response, frame = stream.read()
	if not response:
		await asyncio.sleep(0.5)
		return None

	return frame

async def detect_license_plate(detection_model: YOLO, frame: MatLike, frame_size: int, inference_device: str) -> List[MatLike]:
	detections: Generator = detection_model.predict(frame, imgsz=frame_size, half=True, device=inference_device)

	return extract_license_plate(detections, frame_size)

async def recognize_license_number(recognition_model: PaddleOCR, images: List[MatLike]) -> List[str]:
	if len(images) == 0:
		return []

	return list(filter(None, extract_license_number(images, recognition_model)))

async def send_results(results: List[str], api_endpoint: str) -> None:
	if len(results) == 0:
		return

	print(f"LICENSE NUMBER RECOGNIZER: detect {', '.join(results)}.")
	# for result in results:
	# 	requests.post(api_endpoint, data={"車牌號碼": result, "名稱": "車牌辨識"})
