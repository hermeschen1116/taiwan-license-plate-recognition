import os
import tempfile
from typing import Dict, List

import polars
from dotenv import load_dotenv

from datasets import DatasetDict
from datasets.features.image import Image

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")

data_directory: str = f"{project_root}/datasets/ocr"

with tempfile.TemporaryDirectory() as temp_dir:
	data: List[Dict[str, str]] = [
		{"label": file.split(".")[0], "image": os.path.join(data_directory, file)}
		for file in os.listdir(data_directory)
	]
	polars.DataFrame(data).write_csv(f"{temp_dir}/data.csv")
	dataset = DatasetDict.from_csv({"train": f"{temp_dir}/data.csv"})
	dataset["train"], dataset["validation"] = dataset["train"].train_test_split(test_size=0.2).values()
	dataset["train"], dataset["test"] = dataset["train"].train_test_split(test_size=0.125).values()
	dataset = dataset.cast_column("image", Image(decode=False))

	print(dataset["train"][0])
