import os
import tempfile
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm

from datasets import Image, load_dataset
from taiwan_license_plate_recognition.EasyOCR.trainer.train import train
from taiwan_license_plate_recognition.EasyOCR.trainer.utils import AttrDict
from taiwan_license_plate_recognition.helper import get_num_of_workers

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()

dataset = load_dataset("hermeschen1116/taiwan-license-plate-ocr", num_proc=num_workers)
dataset = dataset.remove_columns(["label_other"])
dataset = dataset.cast_column("image", Image(decode=True))

with tempfile.TemporaryDirectory() as dataset_directory:
	for split in ["train", "validation", "test"]:
		os.makedirs(f"{dataset_directory}/{split}/images")
		for sample in tqdm(dataset[split], desc=f"{split}: ", colour="green"):
			sample["image"].save(f"{dataset_directory}/{split}/images/{sample['label']}.png")
		dataset[split].to_csv(f"{dataset_directory}/{split}/label.csv", num_proc=num_workers)

	options: Dict[str, Any] = {}
	with open(f"{project_root}/src/scripts/ocr/arguments/easyOCR.yaml", "r", encoding="utf8") as file:
		options = AttrDict(yaml.safe_load(file))

	options["character"] = options["number"] + options["symbol"] + options["lang_char"]

	os.makedirs(f"./saved_models/{options['experiment_name']}", exist_ok=True)

	train(options, amp=False)
