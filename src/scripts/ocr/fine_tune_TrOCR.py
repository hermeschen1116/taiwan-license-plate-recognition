import os
from typing import Dict

import evaluate
import torch
from PIL.Image import Resampling
from dotenv import load_dotenv
from transformers import (
	GenerationConfig,
	Seq2SeqTrainer,
	Seq2SeqTrainingArguments,
	TrOCRProcessor,
	VisionEncoderDecoderModel,
)

import wandb
from datasets import Image, load_dataset
from taiwan_license_plate_recognition.Helper import get_num_of_workers, get_torch_device

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()
device: str = get_torch_device()
max_length: int = 64

run = wandb.init(job_type="fine_tune", project="taiwan-license-plate-recognition", group="TrOCR")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", clean_up_tokenization_spaces=True)

dataset = load_dataset("hermeschen1116/taiwan-license-plate-ocr", keep_in_memory=True, num_proc=num_workers)
dataset = dataset.remove_columns(["label_other"])

dataset = dataset.cast_column("image", Image(decode=True))

dataset = dataset.map(
	lambda samples: {"image": [sample.resize((384, 384), resample=Resampling.BILINEAR) for sample in samples]},
	input_columns=["image"],
	batched=True,
)


def encode_image(image):
	return processor(image=image.convert("RGB"), return_tensors="pt").pixel_values.squeeze()


dataset.set_transform(encode_image, columns=["image"], output_all_columns=True)
dataset.set_format("torch", columns=["image"], output_all_columns=True)
dataset = dataset.rename_column("image", "pixel_values")


def encode_label(label):
	encoded_label = processor.tokenizer(label, padding="max_length", max_length=max_length).input_ids
	encoded_label[encoded_label == processor.tokenizer.pad_token_id] = -100
	return encoded_label


dataset = dataset.map(
	lambda samples: {"label": [encode_label(sample.replace("-", "")) for sample in samples]},
	input_columns=["label"],
	batched=True,
	num_proc=num_workers,
).rename_column("label", "labels")
dataset.set_format("torch", columns=["pixel_values", "labels"], output_all_columns=True)

model = VisionEncoderDecoderModel.from_pretrained(
	"DunnBC22/trocr-base-printed_license_plates_ocr", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = max_length
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

trainer_arguments = Seq2SeqTrainingArguments(
	output_dir=f"./checkpoints/{run.name}",
	overwrite_output_dir=True,
	eval_strategy="steps",
	per_device_train_batch_size=4,
	per_device_eval_batch_size=4,
	eval_accumulation_steps=50,
	run_name=run.name,
	eval_delay=500,
	dataloader_num_workers=int(num_workers / 2),
	num_train_epochs=5,
	lr_scheduler_type="reduce_lr_on_plateau",
	logging_steps=25,
	save_steps=25,
	save_total_limit=5,
	load_best_model_at_end=True,
	metric_for_best_model="cer",
	greater_is_better=False,
	bf16=True,
	fp16=False,
	optim="paged_lion_32bit",
	report_to=["wandb"],
	dataloader_pin_memory=True,
	dataloader_persistent_workers=True,
	auto_find_batch_size=True,
	eval_on_start=True,
	sortish_sampler=True,
	predict_with_generate=True,
)

cer_metric = evaluate.load("cer", keep_in_memory=True)


def compute_metrics(eval_prediction) -> Dict[str, float]:
	prediction_ids, label_ids = eval_prediction

	prediction = processor.batch_decode(prediction_ids, skip_special_tokens=True)
	label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
	label = processor.batch_decode(label_ids, skip_special_tokens=True)

	cer_score = cer_metric.compute(predictions=prediction, references=label)

	return {"cer": cer_score}


trainer = Seq2SeqTrainer(
	model=model,
	tokenizer=processor.image_processor,
	args=trainer_arguments,
	compute_metrics=compute_metrics,
	train_dataset=dataset["train"],
	eval_dataset=dataset["validation"],
)

trainer.train()

model = torch.compile(model, dynamic=True, backend="openvino", mode="reduce-overhead")

trainer.evaluate(dataset["test"], metric_key_prefix="test")

model.push_to_hub("taiwan-license-plate-recognition")

generation_config = GenerationConfig(
	decoder_start_token_id=processor.tokenizer.cls_token_id,
	pad_token_id=processor.tokenizer.pad_token_id,
	vocab_size=model.config.decoder.vocab_size,
	eos_token_id=processor.tokenizer.sep_token_id,
	max_length=max_length,
	early_stopping=True,
	no_repeat_ngram_size=3,
	length_penalty=2.0,
	num_beams=4,
)

generation_config.push_to_hub("taiwan-license-plate-recognition")

run.finish()
