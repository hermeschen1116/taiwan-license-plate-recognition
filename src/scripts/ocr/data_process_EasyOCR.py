import os
import random
import shutil
import tempfile
from typing import Dict, List

import polars
from dotenv import load_dotenv

from datasets import load_dataset
from datasets.features.image import Image
from taiwan_license_plate_recognition.helper import get_num_of_workers

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()

data_source: str = f"{project_root}/datasets/ocr"

with tempfile.TemporaryDirectory() as temp_dir:
	files: List[str] = os.listdir(data_source)
	num_files: int = len(files)

	files = random.sample(files, num_files)

	num_train_split: int = int(num_files * 0.7)
	num_validation_split: int = int(num_files * 0.2)
	splits: Dict[str, List[str]] = {
		"train": files[:num_train_split],
		"validation": files[num_train_split : (num_train_split + num_validation_split)],
		"test": files[(num_train_split + num_validation_split) :],
	}

	dataset_path: str = f"{temp_dir}/datasets"
	os.makedirs(dataset_path)
	for split_name in splits.keys():
		os.makedirs(f"{dataset_path}/{split_name}")
		for file in splits[split_name]:
			shutil.copyfile(os.path.join(data_source, file), f"{dataset_path}/{split_name}/{file}")
		metadata: List[Dict[str, str]] = [
			{"file_name": file, "label": file.split(".")[0]} for file in splits[split_name]
		]
		polars.DataFrame(metadata).write_csv(f"{dataset_path}/{split_name}/metadata.csv")

	dataset = load_dataset("imagefolder", data_dir=dataset_path, save_infos=True, num_proc=num_workers)
	dataset = dataset.cast_column("image", Image(decode=False))

	def contain_chinese(s: str) -> bool:
		return s[0:3] in ["台北市", "高雄市", "台灣省", "電動車"]

	dataset = dataset.map(
		lambda samples: {
			"label_other": [label[:3] if contain_chinese(label) else "" for label in samples],
			"label": [label[3:] if contain_chinese(label) else label for label in samples],
		},
		input_columns=["label"],
		batched=True,
		num_proc=num_workers,
	)

	dataset.push_to_hub(
		"taiwan-license-plate-ocr",
		private=True,
		num_shards={"train": 16, "validation": 16, "test": 16},
		embed_external_files=True,
	)
