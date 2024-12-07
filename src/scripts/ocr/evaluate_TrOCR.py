import os

import evaluate
from dotenv import load_dotenv
from optimum.intel import OVModelForVision2Seq, OVWeightQuantizationConfig
from transformers import TrOCRProcessor

import datasets
import wandb
from datasets import load_dataset
from taiwan_license_plate_recognition.Helper import accuracy_metric, get_num_of_workers
from taiwan_license_plate_recognition.PostProcess import remove_non_alphanum
from taiwan_license_plate_recognition.Utils import extract_license_number_trocr

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()
max_length: int = 64

run = wandb.init(job_type="evaluate", project="taiwan-license-plate-recognition", group="TrOCR")

dataset = load_dataset(
	"hermeschen1116/taiwan-license-plate-ocr", split="test", keep_in_memory=True, num_proc=num_workers
)
dataset = dataset.remove_columns(["label_other"])

dataset = dataset.cast_column("image", datasets.Image(decode=True))

dataset = dataset.map(
	lambda samples: {"label": [sample.replace("-", "") for sample in samples]},
	input_columns=["label"],
	batched=True,
	num_proc=num_workers,
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

dataset = dataset.map(
	lambda samples: {"prediction": extract_license_number_trocr(samples, model, processor)},
	input_columns=["image"],
	batched=True,
	batch_size=4,
)

dataset.set_transform(remove_non_alphanum, columns=["predictions"], output_all_columns=True)

cer_score = cer_metric.compute(predictions=dataset["prediction"], references=dataset["label"])
accuracy_score = accuracy_metric(predictions=dataset["prediction"], references=dataset["label"])

run.log({"test/cer": cer_score, "test/accuracy": accuracy_score})

result = wandb.Table(dataframe=dataset.remove_columns(["image"]).to_pandas())
run.log({"evaluation_result": result})

run.finish()
