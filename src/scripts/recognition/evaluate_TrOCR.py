import os
from typing import List

import evaluate
import wandb
from PIL.Image import Image
from dotenv import load_dotenv
from optimum.intel import OVModelForVision2Seq, OVWeightQuantizationConfig
from transformers import TrOCRProcessor

import datasets
from datasets import load_dataset
from taiwan_license_plate_recognition.Helper import get_num_of_workers
from taiwan_license_plate_recognition.recognition.Metrics import accuracy
from taiwan_license_plate_recognition.recognition.PostProcess import validate_license_number

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()
max_length: int = 8

run = wandb.init(job_type="evaluate", project="taiwan-license-plate-recognition", group="TrOCR")

dataset = load_dataset(
	"hermeschen1116/taiwan-license-plate-ocr", split="test", keep_in_memory=True, num_proc=num_workers
)
dataset = dataset.remove_columns(["label_other"])

dataset = dataset.cast_column("image", datasets.Image(decode=True))

dataset = dataset.map(
	lambda samples: {"label": [sample.replace("-", "") for sample in samples]}, input_columns=["label"], batched=True
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", clean_up_tokenization_spaces=True)

quantization_config = OVWeightQuantizationConfig()
ov_config = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": f"{project_root}/.ov_cache"}

model = OVModelForVision2Seq.from_pretrained(
	"hermeschen1116/taiwan-license-plate-recognition",
	export=True,
	ov_config=ov_config,
	quantization_config=quantization_config,
	device="cpu",
)

cer_metric = evaluate.load("cer", keep_in_memory=True)


def extract_license_number(images: List[Image]) -> List[str]:
	if len(images) == 0:
		return []

	encode_image = processor(images, return_tensors="pt").pixel_values

	generated_ids = model.generate(encode_image, max_length=max_length)

	results = processor.batch_decode(generated_ids, skip_special_tokens=True)

	return [str(validate_license_number(result)) for result in results]


dataset = dataset.map(
	lambda samples: {"prediction": extract_license_number(samples)}, input_columns=["image"], batched=True, batch_size=4
)


cer_score = cer_metric.compute(predictions=dataset["prediction"], references=dataset["label"])
accuracy_score = accuracy(predictions=dataset["prediction"], references=dataset["label"])

run.log({"test/cer": cer_score, "test/accuracy": accuracy_score})

result = wandb.Table(dataframe=dataset.remove_columns(["image"]).to_pandas())
run.log({"evaluation_result": result})

run.finish()
