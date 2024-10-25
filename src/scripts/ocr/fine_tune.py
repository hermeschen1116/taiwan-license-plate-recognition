from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from datasets import load_dataset
from license_plate_recognition.helper import get_num_of_workers

num_workers: int = get_num_of_workers()

dataset = load_dataset("gagan3012/IAM", keep_in_memory=True, num_proc=num_workers)

tokenizer = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

base_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
