import os
from typing import List

import cv2
import paddle
import requests
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from ultralytics import YOLO

from taiwan_license_plate_recognition.Helper import get_num_of_workers
from taiwan_license_plate_recognition.Utils import extract_license_number_paddleocr, extract_license_plate

load_dotenv()
paddle.disable_signal_handler()

num_workers: int = get_num_of_workers()

program_name: str = "LICENSE NUMBER RECOGNIZER"
image_size: int = int(os.environ.get("IMAGE_SIZE", 640))
yolo_model_path: str = os.environ.get("YOLO_MODEL_PATH", "")
stream_path: str = os.environ.get("CAMERA_ADDRESS", "")
api: str = os.environ.get("API", "")

yolo_model = YOLO(yolo_model_path, task="obb")

reader = PaddleOCR(lang="en", device="cpu", use_angle_cls=True, max_text_length=8, use_space_char=False, binarize=True)

stream = cv2.VideoCapture(stream_path)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)
stream.set(cv2.CAP_PROP_FPS, 30)

while stream.isOpened():
	response, frame = stream.read()
	if not response:
		continue

	result = yolo_model.predict(frame, imgsz=image_size, device="cpu")
	cropped_images = extract_license_plate(result, image_size)
	license_numbers: List[str] = extract_license_number_paddleocr(cropped_images, reader)
	if len(license_numbers) == 0:
		continue
	for license_number in license_numbers:
		print(f"{program_name}: License number: {license_number}")
		requests.post(api, data={"車牌號碼": license_number, "名稱": "車牌辨識"})
