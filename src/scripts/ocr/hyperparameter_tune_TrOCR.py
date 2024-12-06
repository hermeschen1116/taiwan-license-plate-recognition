import os
from typing import Dict

import evaluate
import torch
from PIL.Image import Resampling
from dotenv import load_dotenv
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel

import wandb
from datasets import Image, load_dataset
from taiwan_license_plate_recognition.Helper import get_num_of_workers, get_torch_device

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()
device: str = get_torch_device()
max_length: int = 64

run = wandb.init(job_type="fine_tune", project="taiwan-license-plate-recognition", group="TrOCR", mode="offline")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", clean_up_tokenization_spaces=True)

dataset = load_dataset("hermeschen1116/taiwan-license-plate-ocr", keep_in_memory=True, num_proc=num_workers)
dataset = dataset.remove_columns(["label_other"])
print(dataset)

dataset = dataset.cast_column("image", Image(decode=True))

dataset = dataset.map(
	lambda samples: {"image": [sample.resize((384, 384), resample=Resampling.BILINEAR) for sample in samples]},
	input_columns=["image"],
	batched=True,
	num_proc=num_workers,
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
	lambda samples: {"label": [encode_label(sample) for sample in samples]},
	input_columns=["label"],
	batched=True,
	num_proc=num_workers,
).rename_column("label", "labels")
dataset.set_format("torch", columns=["pixel_values", "labels"], output_all_columns=True)


def model_init():
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

	return model


trainer_arguments = Seq2SeqTrainingArguments(
	output_dir="./checkpoints",
	overwrite_output_dir=True,
	eval_strategy="steps",
	per_device_train_batch_size=4,
	per_device_eval_batch_size=4,
	eval_accumulation_steps=50,
	torch_empty_cache_steps=100,
	eval_delay=500,
	# learning_rate=2e-4,
	# weight_decay=0.001,
	# max_grad_norm=0.3,
	num_train_epochs=1,
	lr_scheduler_type="linear",
	# warmup_ratio=0.003,
	logging_steps=25,
	save_steps=25,
	save_total_limit=5,
	load_best_model_at_end=True,
	metric_for_best_model="cer",
	bf16=True,
	fp16=False,
	dataloader_num_workers=num_workers,
	optim="paged_adamw_32bit",
	group_by_length=True,
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

	return {"cer": cer_metric.compute(predictions=prediction, references=label)}


trainer = Seq2SeqTrainer(
	model_init=model_init,
	tokenizer=processor.image_processor,
	args=trainer_arguments,
	compute_metrics=compute_metrics,
	train_dataset=dataset["train"],
	eval_dataset=dataset["validation"],
)

trainer.hyperparameter_search()

wandb.finish()

exit(0)
