import logging
import os
from typing import List

import requests
from dotenv import load_dotenv
from optimum.intel import OVModelForVision2Seq, OVWeightQuantizationConfig
from transformers import TrOCRProcessor
from ultralytics import YOLO

from taiwan_license_plate_recognition.Utils import extract_license_number, extract_license_plate

load_dotenv()

program_name: str = "LICENSE NUMBER RECOGNIZER"
yolo_model_path: str = os.environ.get("YOLO_MODEL_PATH", "")
logging.info(f"{program_name}: YOLO_MODEL_PATH: {yolo_model_path}")
ocr_processor_path: str = os.environ.get("OCR_PROCESSOR_PATH", "microsoft/trocr-base-printed")
logging.info(f"{program_name}: OCR_PROCESSOR_PATH: {ocr_processor_path}")
ocr_model_path: str = os.environ.get("OCR_MODEL_PATH", ocr_processor_path)
logging.info(f"{program_name}: OCR_MODEL_PATH: {ocr_model_path}")
stream_path: str = os.environ.get("CAMERA_ADDRESS", "")
logging.info(f"{program_name}: CAMERA_ADDRESS: {stream_path}")
api: str = os.environ.get("API", "")

logging.info(f"{program_name}: Initialize YOLO model.")
yolo_model = YOLO(yolo_model_path, task="obb")

logging.info(f"{program_name}: Initialize TROCR processor.")
ocr_processor = TrOCRProcessor.from_pretrained(ocr_processor_path, clean_up_tokenization_spaces=True)

quantization_config = OVWeightQuantizationConfig()
ov_config = {"PERFORMANCE_HINT": "LATENCY"}

logging.info(f"{program_name}: Initialize TROCR model.")
ocr_model = OVModelForVision2Seq.from_pretrained(
	ocr_model_path, export=True, ov_config=ov_config, quantization_config=quantization_config, device="cpu"
)

logging.info(f"{program_name}: Start detecting")
for result in yolo_model.predict(stream_path, stream=True, stream_buffer=True, device="cpu"):
	cropped_images = extract_license_plate(result)
	license_numbers: List[str] = extract_license_number(cropped_images, ocr_model, ocr_processor)
	logging.info(f"{program_name}: License number: {', '.join(license_numbers)}")
	print(f"{program_name}: License number: {', '.join(license_numbers)}")
	requests.post(api, data={"名稱": "車牌辨識", "車牌號碼": license_numbers[0]})
	# subprocess.run(["curl", "-d", f"名稱=車牌辨識&車牌號碼={license_numbers[0]}", api])
