from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from datasets import load_dataset
from taiwan_license_plate_recognition.helper import get_num_of_workers

num_workers: int = get_num_of_workers()

dataset = load_dataset("gagan3012/IAM", keep_in_memory=True, num_proc=num_workers)["train"].remove_columns(["label"])

tokenizer = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

base_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
