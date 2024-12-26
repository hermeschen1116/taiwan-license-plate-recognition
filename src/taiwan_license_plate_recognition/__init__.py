import os
from typing import Generator, List

import cv2
import paddle
import requests
from cv2.typing import MatLike
from paddleocr import PaddleOCR
from ultralytics import YOLO

from taiwan_license_plate_recognition.detection import extract_license_plate
from taiwan_license_plate_recognition.recognition import extract_license_number


def load_detection_model() -> YOLO:
	detection_model_path: str = os.environ.get("DETECTION_MODEL_PATH", "")
	if not detection_model_path:
		raise ValueError("LICENSE NUMBER RECOGNIZER: DETECTION_MODEL_PATH not set.")

	detection_model: YOLO = YOLO(detection_model_path, task="obb")
	print("LICENSE NUMBER RECOGNIZER: detection model loaded.")

	return detection_model


def load_recognition_model(*args, **kwargs) -> PaddleOCR:
	paddle.disable_signal_handler()

	recognition_model: PaddleOCR = PaddleOCR(*args, **kwargs)
	print("LICENSE NUMBER RECOGNIZER: recognition model loaded.")

	return recognition_model


def initialize_stream(frame_size: int) -> cv2.VideoCapture:
	# stream_source: str = os.environ.get("STREAM_SOURCE", "")
	stream_source: int = 1
	if not stream_source:
		raise ValueError("LICENSE NUMBER RECOGNIZER: STREAM_SOURCE not set.")

	stream: cv2.VideoCapture = cv2.VideoCapture(stream_source)
	print("LICENSE NUMBER RECOGNIZER: stream initialized.")

	stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
	stream.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size)
	stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size)
	stream.set(cv2.CAP_PROP_FPS, 1)

	return stream


def process_image(
	detection_model: YOLO, recognition_model: PaddleOCR, frame: MatLike, frame_size: int, inference_device: str
) -> List[str]:
	print("LICENSE NUMBER RECOGNIZER: detecting.")

	detections: Generator = detection_model.predict(frame, imgsz=frame_size, half=True, device=inference_device)

	images: List[MatLike] = extract_license_plate(detections, frame_size)

	if not images:
		return []

	return list(filter(None, extract_license_number(images, recognition_model)))


def send_results(results: List[str], api_endpoint: str) -> None:
	print("LICENSE NUMBER RECOGNIZER: sending.")
	if len(results) == 0:
		return

	for result in results:
		print(f"LICENSE NUMBER RECOGNIZER: detect {result}.")
		# requests.post(api_endpoint, data={"車牌號碼": result, "名稱": "車牌辨識"})
