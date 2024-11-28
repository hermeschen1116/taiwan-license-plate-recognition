import os
import tempfile
from pprint import pprint

from dotenv import load_dotenv
from tqdm.auto import tqdm

from datasets import Image, load_dataset
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

		pprint(os.listdir(f"{dataset_directory}/{split}/"))
		pprint(os.listdir(f"{dataset_directory}/{split}/images")[0:10])
