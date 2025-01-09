import os
import shutil
from typing import List

import cv2
import datasets
import numpy
import paddle
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv

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

data_dir: str = f"{project_root}/datasets/rec"

if os.path.exists(data_dir):
	shutil.rmtree(data_dir)
else:
	os.makedirs(data_dir)
	os.makedirs(f"{data_dir}/rec_train_images")
	os.makedirs(f"{data_dir}/rec_test_images")

with open(f"{data_dir}/train_rec_label.txt", "a") as file:
	annotations: List[str] = [
		f"rec_train_images/{sample['path']}\t{sample['label']}\n".replace("'", '"') for sample in dataset["train"]
	]
	file.writelines(annotations)

for sample in dataset["train"]:
	image = Image.fromarray(sample["image"], "RGB")
	image.save(f"{data_dir}/rec_train_images/{sample['path']}")

with open(f"{data_dir}/test_rec_label.txt", "a") as file:
	annotations: List[str] = [
		f"rec_test_images/{sample['path']}\t{sample['label']}\n".replace("'", '"') for sample in dataset["validation"]
	]
	file.writelines(annotations)

for sample in dataset["validation"]:
	image = Image.fromarray(sample["image"], "RGB")
	image.save(f"{data_dir}/rec_test_images/{sample['path']}")
