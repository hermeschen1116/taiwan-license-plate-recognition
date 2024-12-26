import os
from typing import List

import cv2
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from ultralytics import YOLO

from taiwan_license_plate_recognition import (
	initialize_stream,
	load_detection_model,
	load_recognition_model,
	process_image,
	send_results,
)

load_dotenv()

inference_device: str = os.environ.get("INFERENCE_DEVICE", "cpu")
frame_size: int = int(os.environ.get("FRAME_SIZE", 640))
api_endpoint: str = os.environ.get("API_ENDPOINT", "")

detection_model: YOLO = load_detection_model()

recognition_model: PaddleOCR = load_recognition_model(
	lang="en", binarize=True, use_angle_cls=True, max_text_length=8, use_space_char=False, device=inference_device
)

stream: cv2.VideoCapture = initialize_stream(frame_size)

while stream.isOpened():
	response, frame = stream.read()
	if not response:
		print("LICENSE NUMBER RECOGNIZER: fail to get frame.")
		continue

	results: List[str] = process_image(detection_model, recognition_model, frame, frame_size, inference_device)

	send_results(results, api_endpoint)
