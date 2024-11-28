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

dataset = load_dataset("hermeschen1116/taiwan-license-plate-ocr", keep_in_memory=True, num_proc=num_workers)
dataset = dataset.remove_columns(["label_other"])
dataset = dataset.cast_column("image", Image(decode=True))
dataset = dataset.map(
	lambda samples: {"filename": [f"{sample}.jpg" for sample in samples]},
	input_columns=["label"],
	batched=True,
	num_proc=num_workers,
)
dataset = dataset.rename_column("label", "words")
print(dataset.column_names)

with tempfile.TemporaryDirectory() as run_directory:
	for split in ["train", "validation", "test"]:
		os.makedirs(f"{run_directory}/{split}/images")
		for sample in tqdm(dataset[split], desc=f"{split}: ", colour="green"):
			sample["image"].save(f"{run_directory}/{split}/images/{sample['words']}.png")
		dataset[split].remove_columns(["image"]).to_csv(
			f"{run_directory}/{split}/images/labels.csv", num_proc=num_workers
		)

	options: Dict[str, Any] = {}
	with open(f"{project_root}/src/scripts/ocr/arguments/easyOCR.yaml", "r", encoding="utf8") as file:
		options = AttrDict(yaml.safe_load(file))

	options["train_data"] = f"{run_directory}/train"
	options["valid_data"] = f"{run_directory}/validation"

	options["character"] = options["number"] + options["symbol"] + options["lang_char"]

	os.makedirs(f"./saved_models/{options['experiment_name']}", exist_ok=True)

	train(options, amp=False)
