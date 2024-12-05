import os

import torch
import wandb
from PIL.Image import Resampling
from dotenv import load_dotenv
from optimum.intel import OVModelForVision2Seq
from transformers import TrOCRProcessor

from datasets import Image, load_dataset
from taiwan_license_plate_recognition.helper import get_num_of_workers, get_torch_device

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

model = OVModelForVision2Seq.from_pretrained(
	"DunnBC22/trocr-base-printed_license_plates_ocr", export=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
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
