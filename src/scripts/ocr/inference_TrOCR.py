import os

from PIL import Image
from dotenv import load_dotenv
from optimum.intel import OVModelForVision2Seq
from transformers import TrOCRProcessor
from transformers.pipelines import ImageToTextPipeline

from taiwan_license_plate_recognition.helper import get_num_of_workers, get_torch_device

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()
device: str = get_torch_device()
max_length: int = 64

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", clean_up_tokenization_spaces=True)

test_image_path: str = f"{project_root}/datasets/ocr/高雄市ZS-0786.jpg"

image = Image.open(test_image_path)

model = OVModelForVision2Seq.from_pretrained(
	"hermeschen1116/taiwan-license-plate-recognition", export=True, device="cpu"
)

recognizer = ImageToTextPipeline(
	model=model,
	tokenizer=processor.tokenizer,
	image_processor=processor,
	framework="pt",
	task="image-to-text",
	num_workers=num_workers,
	torch_dtype="auto",
)

print(f"Loaded Image: {recognizer(image)}")
print(f"Image File: {recognizer(test_image_path)}")
