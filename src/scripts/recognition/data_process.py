import os
import shutil
import tempfile
from typing import Dict, List

import polars
from dotenv import load_dotenv

from datasets import DatasetDict
from datasets.features.image import Image
from taiwan_license_plate_recognition.Utils import get_num_of_workers

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()

data_source: str = f"{project_root}/datasets/ocr"

with tempfile.TemporaryDirectory() as temp_dir:
	dataset_path: str = f"{temp_dir}/data_source"
	shutil.copytree(data_source, dataset_path, dirs_exist_ok=True)

	data: List[Dict[str, str]] = [
		{"label": file.split(".")[0], "image": os.path.join(dataset_path, file)} for file in os.listdir(dataset_path)
	]
	polars.DataFrame(data).write_csv(f"{temp_dir}/data.csv")

	dataset = DatasetDict.from_csv({"train": f"{temp_dir}/data.csv"})
	dataset["train"], dataset["validation"] = dataset["train"].train_test_split(test_size=0.2).values()
	dataset["train"], dataset["test"] = dataset["train"].train_test_split(test_size=0.125).values()
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
		"taiwan-license-plate-ocr", num_shards={"train": 8, "validation": 8, "test": 8}, embed_external_files=True
	)
