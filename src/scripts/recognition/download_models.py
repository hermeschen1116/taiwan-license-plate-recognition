import os
import tarfile

import wget
from dotenv import load_dotenv

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
model_dir: str = f"{project_root}/models"

if not os.path.isdir(model_dir):
	os.makedirs(model_dir)

print("Downloading Detection Model.......")
wget.download(
	url="https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_distill_train.tar",
	out=f"{model_dir}/en_PP-OCRv3_det_slim_distill_train.tar",
)

with tarfile.open(f"{model_dir}/en_PP-OCRv3_det_slim_distill_train.tar", "r") as tar:
	print("Extracting Model Files.......")
	tar.extractall(f"{model_dir}/")

print("Downloading Recognition Model......")
wget.download(
	url="https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_train.tar",
	out=f"{model_dir}/en_PP-OCRv3_rec_slim_train.tar",
)

with tarfile.open(f"{model_dir}/en_PP-OCRv3_rec_slim_train.tar", "r") as tar:
	print("Extracting Model Files.......")
	tar.extractall(f"{model_dir}/")
