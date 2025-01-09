import os
import shutil
from typing import List

import cv2
import numpy
import paddle
from PIL import Image
from dotenv import load_dotenv
from tqdm.auto import tqdm

import datasets
from datasets import load_dataset
from taiwan_license_plate_recognition.Utils import get_num_of_workers

paddle.disable_signal_handler()
load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()

dataset = load_dataset("hermeschen1116/taiwan-license-plate-ocr", num_proc=num_workers)

dataset = dataset.cast_column("image", datasets.Image(decode=True))


def encode_image(image) -> numpy.ndarray:
	cv2_image = cv2.cvtColor(numpy.asarray(image, dtype=numpy.uint8), cv2.COLOR_RGB2BGR)

	return cv2_image


dataset.set_transform(encode_image, columns=["image"], output_all_columns=True)
dataset.set_format("numpy", columns=["image"], output_all_columns=True)

data_dir: str = f"{project_root}/datasets/det"

if os.path.exists(data_dir):
	shutil.rmtree(data_dir)
else:
	os.makedirs(data_dir)
	os.makedirs(f"{data_dir}/det_train_images")
	os.makedirs(f"{data_dir}/det_test_images")

with open(f"{data_dir}/train_det_label.txt", "a") as file:
	annotations: List[str] = [
		f"det_train_images/{sample['path']}\t{sample['annotation']}\n".replace("'", '"')
		for sample in tqdm(dataset["train"], desc="Detection Dataset (train): ", colour="green")
	]
	file.writelines(annotations)

for sample in tqdm(dataset["train"], desc="Detection Dataset (train): ", colour="blue"):
	image = Image.fromarray(sample["image"], "RGB")
	image.save(f"{data_dir}/det_train_images/{sample['path']}")

with open(f"{data_dir}/test_det_label.txt", "a") as file:
	annotations: List[str] = [
		f"det_test_images/{sample['path']}\t{sample['annotation']}\n".replace("'", '"')
		for sample in tqdm(dataset["validation"], desc="Detection Dataset (test): ", colour="green")
	]
	file.writelines(annotations)

for sample in tqdm(dataset["validation"], desc="Detection Dataset (test): ", colour="blue"):
	image = Image.fromarray(sample["image"], "RGB")
	image.save(f"{data_dir}/det_test_images/{sample['path']}")
