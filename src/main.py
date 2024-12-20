import os
import time
from typing import Generator, List

import cv2
import paddle
import requests
from cv2.typing import MatLike
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from ultralytics import YOLO

from taiwan_license_plate_recognition.Utils import get_num_of_workers
from taiwan_license_plate_recognition.detection import extract_license_plate
from taiwan_license_plate_recognition.recognition import extract_license_number

load_dotenv()
paddle.disable_signal_handler()

project_root: str = os.environ.get("PROJECT_ROOT", "")
program_name: str = "LICENSE NUMBER RECOGNIZER"

inference_device: str = os.environ.get("INFERENCE_DEVICE", "cpu")
num_workers: int = int(os.environ.get("NUM_WORKERS", get_num_of_workers()))
frame_size: int = int(os.environ.get("FRAME_SIZE", 640))
stream_path: str = os.environ.get("STREAM_SOURCE", "")
detection_model_path: str = os.environ.get("DETECTION_MODEL_PATH", "")
api_endpoint: str = os.environ.get("API_ENDPOINT", "")

yolo_model: YOLO = YOLO(detection_model_path, task="obb")

reader: PaddleOCR = PaddleOCR(
	lang="en",
	device=inference_device,
	use_angle_cls=True,
	max_text_length=8,
	total_process_num=num_workers,
	use_mp=True,
	use_space_char=False,
	binarize=True,
)

stream: cv2.VideoCapture = cv2.VideoCapture(stream_path)
stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
stream.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size)
stream.set(cv2.CAP_PROP_FPS, 1)

while stream.isOpened():
	key: int = cv2.waitKey(90)
	if key == ord("q") or key == 27:
		stream.release()
		break

	response, frame = stream.read()
	if not response:
		continue
	cv2.imwrite(f"{project_root}/log/{time.time()}.png", frame)

	detections: Generator = yolo_model.predict(frame, imgsz=frame_size, half=True, device=inference_device)
	cropped_images: List[MatLike] = extract_license_plate(detections, frame_size)
	license_numbers: List[str] = list(filter(None, extract_license_number(cropped_images, reader)))
	if len(license_numbers) == 0:
		continue
	for license_number in license_numbers:
		print(f"{program_name}: License number: {license_number}")
		requests.post(api_endpoint, data={"車牌號碼": license_number, "名稱": "車牌辨識"})
