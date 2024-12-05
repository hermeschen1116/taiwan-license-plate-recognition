import os
import time

import evaluate
from PIL import Image
from dotenv import load_dotenv
from optimum.intel import OVModelForVision2Seq, OVWeightQuantizationConfig
from transformers import TrOCRProcessor
from transformers.pipelines import ImageToTextPipeline

import datasets
import wandb
from datasets import load_dataset
from taiwan_license_plate_recognition.helper import get_num_of_workers

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()
max_length: int = 64

run = wandb.init(job_type="evaluate", project="taiwan-license-plate-recognition", group="TrOCR")

test_image_path: str = f"{project_root}/datasets/ocr/高雄市ZS-0786.jpg"

image = Image.open(test_image_path)

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
ov_config = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "./ov_cache"}

model = OVModelForVision2Seq.from_pretrained(
	"hermeschen1116/taiwan-license-plate-recognition",
	export=True,
	ov_config=ov_config,
	quantization_config=quantization_config,
	device="cpu",
)

recognizer = ImageToTextPipeline(
	model=model,
	tokenizer=processor.tokenizer,
	image_processor=processor,
	framework="pt",
	task="image-to-text",
	num_workers=num_workers,
	device="cpu",
	torch_dtype="auto",
)

start_time = time.time()
result = recognizer(image)
end_time = time.time()
print(f"Loaded Image: {result}, execution time: {end_time - start_time}s")
start_time = time.time()
result = recognizer(test_image_path)
end_time = time.time()
print(f"Image File: {result}, execution time: {end_time - start_time}s")

cer_metric = evaluate.load("cer", keep_in_memory=True)
accuracy_metric = evaluate.load("exact_match", keep_in_memory=True)

dataset = dataset.map(lambda sample: {"prediction": recognizer(sample)[0]["generated_text"]}, input_columns=["image"])

cer_score = cer_metric.compute(predictions=dataset["prediction"], references=dataset["label"])
accuracy_score = cer_metric.compute(predictions=dataset["prediction"], references=dataset["label"])

run.log({"test/cer": cer_score, "test/accuracy": accuracy_score})

result = wandb.Table(dataframe=dataset.remove_columns(["image"]).to_pandas())
run.log({"evaluation_result": result})

run.finish()
